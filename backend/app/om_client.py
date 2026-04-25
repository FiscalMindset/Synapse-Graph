from __future__ import annotations

import asyncio
import base64
import logging
import re
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import uuid

from metadata.generated.schema.api.classification.createClassification import (
    CreateClassificationRequest,
)
from metadata.generated.schema.api.classification.createTag import CreateTagRequest
from metadata.generated.schema.api.data.createDatabase import CreateDatabaseRequest
from metadata.generated.schema.api.data.createDatabaseSchema import CreateDatabaseSchemaRequest
from metadata.generated.schema.api.data.createTable import CreateTableRequest
from metadata.generated.schema.api.lineage.addLineage import AddLineageRequest
from metadata.generated.schema.api.services.createDatabaseService import CreateDatabaseServiceRequest
from metadata.generated.schema.entity.data.table import Column, DataType
from metadata.generated.schema.entity.services.connections.database.common.basicAuth import BasicAuth
from metadata.generated.schema.entity.services.connections.database.mysqlConnection import (
    MysqlConnection,
)
from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
    OpenMetadataConnection,
)
from metadata.generated.schema.entity.services.databaseService import DatabaseConnection
from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
    OpenMetadataJWTClientConfig,
)
from metadata.generated.schema.type.entityLineage import ColumnLineage, EntitiesEdge, LineageDetails
from metadata.generated.schema.type.entityReference import EntityReference
from metadata.ingestion.ometa.ometa_api import OpenMetadata

from .inference import ModelTopology, TokenStepCapture, HeadMask

LOGGER = logging.getLogger(__name__)

_HEAD_NAME_PATTERN = re.compile(r"head[_:\s-]*(\d+)", re.IGNORECASE)


class OpenMetadataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYNAPSE_",
        env_file=[".env", "backend/.env", "../.env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openmetadata_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SYNAPSE_OPENMETADATA_ENABLED", "OPENMETADATA_ENABLED"),
    )
    openmetadata_host: str = Field(
        default="http://127.0.0.1:8585/api",
        validation_alias=AliasChoices("SYNAPSE_OPENMETADATA_HOST", "OPENMETADATA_HOST"),
    )
    openmetadata_auth_provider: str = "openmetadata"
    openmetadata_jwt_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SYNAPSE_OPENMETADATA_JWT_TOKEN",
            "OPENMETADATA_JWT_TOKEN",
            "OPENMETADATA_API_KEY",
        ),
    )
    openmetadata_email: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SYNAPSE_OPENMETADATA_EMAIL",
            "OPENMETADATA_EMAIL",
            "OPENMETADATA_USERNAME",
        ),
    )
    openmetadata_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNAPSE_OPENMETADATA_PASSWORD", "OPENMETADATA_PASSWORD"),
    )
    openmetadata_request_timeout_seconds: float = 15.0
    openmetadata_token_refresh_skew_seconds: float = 60.0

    openmetadata_service_name: str = "Synapse_Neural_Service"
    openmetadata_schema_name: str = "Transformer_Graph"
    openmetadata_prompt_table_name: str = "Prompt_Ingress"
    openmetadata_response_table_name: str = "Response_Egress"
    openmetadata_classification_name: str = "SynapseQuarantine"
    openmetadata_defective_tag_name: str = "DEFECTIVE"
    openmetadata_lineage_top_heads_per_layer: int = 2
    # When set, incoming OpenMetadata webhooks must include this secret in the
    # `X-OpenMetadata-Secret` header to be accepted by the neural proxy.
    openmetadata_webhook_secret: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SYNAPSE_OPENMETADATA_WEBHOOK_SECRET", "OPENMETADATA_WEBHOOK_SECRET"),
    )

    @model_validator(mode="after")
    def normalize_openmetadata_host(self) -> OpenMetadataSettings:
        self.openmetadata_host = _normalize_openmetadata_host(self.openmetadata_host)
        return self


class TableBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str
    table_fqn: str
    table_id: str
    layer_index: int | None = None
    column_fqns: dict[str, str] = Field(default_factory=dict)


class NeuralCatalogBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    service_fqn: str
    database_fqn: str
    schema_fqn: str
    prompt_table: TableBinding
    response_table: TableBinding
    layer_tables: list[TableBinding] = Field(default_factory=list)
    classification_fqn: str
    defective_tag_fqn: str

    def layer_binding(self, layer_index: int) -> TableBinding | None:
        for table in self.layer_tables:
            if table.layer_index == layer_index:
                return table
        return None


class OpenMetadataTokenBundle(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    access_token: str = Field(alias="accessToken")
    refresh_token: str | None = Field(default=None, alias="refreshToken")
    token_type: str = Field(default="Bearer", alias="tokenType")
    expiry_duration: int | None = Field(default=None, alias="expiryDuration")

    @property
    def expires_at(self) -> datetime | None:
        if self.expiry_duration is None:
            return None
        return datetime.fromtimestamp(self.expiry_duration / 1000, tz=timezone.utc)

    @property
    def authorization_header(self) -> str:
        return f"{self.token_type} {self.access_token}"


class OpenMetadataNeuralMapper:
    def __init__(self, settings: OpenMetadataSettings | None = None) -> None:
        self.settings = settings or OpenMetadataSettings()
        self._metadata: OpenMetadata | None = None
        self._catalog_cache: dict[str, NeuralCatalogBinding] = {}
        self._sync_client = httpx.Client(
            base_url=self.settings.openmetadata_host.rstrip("/"),
            timeout=self.settings.openmetadata_request_timeout_seconds,
            headers={"Content-Type": "application/json"},
        )
        self._client = httpx.AsyncClient(
            base_url=self.settings.openmetadata_host.rstrip("/"),
            timeout=self.settings.openmetadata_request_timeout_seconds,
            headers=self._build_headers(),
        )
        self._token_lock = Lock()
        self._token_bundle: OpenMetadataTokenBundle | None = None

    async def close(self) -> None:
        self._sync_client.close()
        await self._client.aclose()

    async def is_available(self) -> bool:
        if not self.settings.openmetadata_enabled:
            return False

        try:
            response = await self._client.get("/v1/system/version")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def ensure_catalog(self, topology: ModelTopology) -> NeuralCatalogBinding | None:
        if not self.settings.openmetadata_enabled:
            return None

        cached = self._catalog_cache.get(topology.model_name)
        if cached is not None:
            return cached

        binding = await asyncio.to_thread(self._ensure_catalog_sync, topology)
        self._catalog_cache[topology.model_name] = binding
        return binding

    async def sync_defective_heads(
        self,
        catalog: NeuralCatalogBinding | None,
    ) -> set[tuple[int, int]]:
        if not self.settings.openmetadata_enabled or catalog is None:
            return set()

        defective_heads: set[tuple[int, int]] = set()
        for layer_table in catalog.layer_tables:
            table_payload = await self._fetch_table_payload(layer_table.table_fqn)
            if not table_payload:
                continue

            if _has_defective_tag(
                table_payload.get("tags", []),
                self.settings.openmetadata_defective_tag_name,
            ):
                defective_heads.update(
                    {
                        (layer_table.layer_index or 0, parsed_head_index)
                        for column_name in layer_table.column_fqns
                        if column_name.startswith("Head_")
                        for parsed_head_index in [_parse_head_index(column_name)]
                        if parsed_head_index is not None
                    }
                )

            for column_payload in table_payload.get("columns", []):
                if not _has_defective_tag(
                    column_payload.get("tags", []),
                    self.settings.openmetadata_defective_tag_name,
                ):
                    continue
                head_index = _parse_head_index(column_payload.get("name") or column_payload.get("displayName"))
                if head_index is None or layer_table.layer_index is None:
                    continue
                defective_heads.add((layer_table.layer_index, head_index))

        return defective_heads

    async def apply_defective_tags(
        self,
        catalog: NeuralCatalogBinding,
        pairs: set[tuple[int, int]],
        reason: str | None = None,
    ) -> list[HeadMask]:
        """
        Apply the defective/quarantine tag in OpenMetadata for the provided
        set of (layer_index, head_index) pairs. Attempts to tag the specific
        column representing the head; falls back to tagging the layer table.

        Returns a list of HeadMask objects that were successfully applied.
        """
        if not self.settings.openmetadata_enabled or catalog is None:
            return []

        applied: list[HeadMask] = []

        for layer_index, head_index in sorted(pairs):
            layer_binding = catalog.layer_binding(layer_index)
            if layer_binding is None:
                continue

            column_name = f"Head_{head_index + 1}"
            table_fqn = layer_binding.table_fqn

            # Prefer column-level tagging when the column exists in the catalog binding
            if column_name in layer_binding.column_fqns:
                path = f"/v1/tables/name/{quote(table_fqn, safe='')}/columns/{quote(column_name, safe='')}/tags"
                payload = {"tagFQN": catalog.defective_tag_fqn}
                try:
                    attempt = 0
                    response = None
                    while attempt < 3:
                        try:
                            response = await self._request_with_managed_auth("POST", path, json=payload)
                            response.raise_for_status()
                            break
                        except Exception:
                            attempt += 1
                            await asyncio.sleep(0.25 * (2 ** attempt))
                    if response is None:
                        raise RuntimeError("Column tag request failed after retries")
                    applied.append(
                        HeadMask(
                            layer_index=layer_index,
                            layer_name=f"Layer_{layer_index + 1}",
                            head_index=head_index,
                            head_name=column_name,
                            reason=reason,
                        )
                    )
                    continue
                except Exception:
                    # Fall through to attempt table-level tagging
                    LOGGER.exception("Column-level tagging failed for %s.%s", table_fqn, column_name)

            # Fallback: tag the whole layer table if column tagging is not available
            try:
                table_path = f"/v1/tables/name/{quote(table_fqn, safe='')}/tags"
                payload = {"tagFQN": catalog.defective_tag_fqn}
                attempt = 0
                response = None
                while attempt < 3:
                    try:
                        response = await self._request_with_managed_auth("POST", table_path, json=payload)
                        response.raise_for_status()
                        break
                    except Exception:
                        attempt += 1
                        await asyncio.sleep(0.25 * (2 ** attempt))
                if response is None:
                    raise RuntimeError("Table tag request failed after retries")
                applied.append(
                    HeadMask(
                        layer_index=layer_index,
                        layer_name=f"Layer_{layer_index + 1}",
                        head_index=head_index,
                        head_name=column_name,
                        reason=reason,
                    )
                )
            except Exception:
                LOGGER.exception("Failed to apply defective tag for table %s", table_fqn)

        return applied

    async def ingest_step(
        self,
        catalog: NeuralCatalogBinding | None,
        session_id: str,
        prompt: str,
        step: TokenStepCapture,
    ) -> None:
        if not self.settings.openmetadata_enabled or catalog is None:
            return
        await asyncio.to_thread(self._ingest_step_sync, catalog, session_id, prompt, step)

    def _ensure_catalog_sync(self, topology: ModelTopology) -> NeuralCatalogBinding:
        metadata = self._metadata_client()

        service_entity = metadata.create_or_update(
            data=CreateDatabaseServiceRequest(
                name=self.settings.openmetadata_service_name,
                displayName="Synapse Neural Service",
                serviceType="Mysql",
                description=(
                    "Synthetic OpenMetadata database service used by Synapse-Graph to "
                    "represent transformer internals as governed lineage assets."
                ),
                connection=DatabaseConnection(
                    config=MysqlConnection(
                        username="synapse",
                        authType=BasicAuth(password="synapse"),
                        hostPort="localhost:3306",
                    )
                ),
            )
        )

        database_name = _sanitize_name(topology.model_name)
        database_entity = metadata.create_or_update(
            data=CreateDatabaseRequest(
                name=database_name,
                displayName=topology.model_name,
                service=_fqn_value(service_entity.fullyQualifiedName),
                description=(
                    "Synthetic database entity representing the active local model inside "
                    "Synapse-Graph."
                ),
            )
        )

        schema_entity = metadata.create_or_update(
            data=CreateDatabaseSchemaRequest(
                name=self.settings.openmetadata_schema_name,
                database=_fqn_value(database_entity.fullyQualifiedName),
                description=(
                    "Transformer layer schema. Each table in this schema represents one "
                    "attention layer and each column anchors a head-level lineage edge."
                ),
            )
        )

        self._bootstrap_governance(metadata)

        prompt_table = self._upsert_table_sync(
            metadata=metadata,
            schema_fqn=_fqn_value(schema_entity.fullyQualifiedName),
            table_name=self.settings.openmetadata_prompt_table_name,
            description="Synthetic ingress table representing the raw user prompt.",
            columns=[
                Column(
                    name="Prompt_Text",
                    displayName="Prompt_Text",
                    dataType=DataType.STRING,
                    description="The raw prompt text received by Synapse-Graph.",
                ),
                Column(
                    name="Prompt_Token_Count",
                    displayName="Prompt_Token_Count",
                    dataType=DataType.INT,
                    description="The tokenized length of the prompt entering the model.",
                ),
            ],
        )

        response_table = self._upsert_table_sync(
            metadata=metadata,
            schema_fqn=_fqn_value(schema_entity.fullyQualifiedName),
            table_name=self.settings.openmetadata_response_table_name,
            description="Synthetic egress table representing the emitted assistant response.",
            columns=[
                Column(
                    name="Response_Text",
                    displayName="Response_Text",
                    dataType=DataType.STRING,
                    description="The final response text produced by the generation backend.",
                ),
            ],
        )

        layer_tables: list[TableBinding] = []
        for layer in topology.layers:
            layer_table = self._upsert_table_sync(
                metadata=metadata,
                schema_fqn=_fqn_value(schema_entity.fullyQualifiedName),
                table_name=layer.layer_name,
                description=(
                    f"Synthetic transformer layer table for {layer.layer_name}. "
                    f"Each column maps to one attention head."
                ),
                columns=[
                    Column(
                        name=f"Head_{head_index + 1}",
                        displayName=f"Head_{head_index + 1}",
                        dataType=DataType.FLOAT,
                        description=(
                            f"Attention head {head_index + 1} activation anchor in "
                            f"{layer.layer_name}."
                        ),
                    )
                    for head_index in range(layer.head_count)
                ],
                layer_index=layer.layer_index,
            )
            layer_tables.append(layer_table)

        return NeuralCatalogBinding(
            model_name=topology.model_name,
            service_fqn=_fqn_value(service_entity.fullyQualifiedName),
            database_fqn=_fqn_value(database_entity.fullyQualifiedName),
            schema_fqn=_fqn_value(schema_entity.fullyQualifiedName),
            prompt_table=prompt_table,
            response_table=response_table,
            layer_tables=layer_tables,
            classification_fqn=self.settings.openmetadata_classification_name,
            defective_tag_fqn=(
                f"{self.settings.openmetadata_classification_name}."
                f"{self.settings.openmetadata_defective_tag_name}"
            ),
        )

    def _bootstrap_governance(self, metadata: OpenMetadata) -> None:
        try:
            metadata.create_or_update(
                data=CreateClassificationRequest(
                    name=self.settings.openmetadata_classification_name,
                    description=(
                        "Governance classification for neural heads that must be isolated "
                        "or zeroed during inference."
                    ),
                )
            )
            metadata.create_or_update(
                data=CreateTagRequest(
                    classification=self.settings.openmetadata_classification_name,
                    name=self.settings.openmetadata_defective_tag_name,
                    description=(
                        "Marks a transformer head as defective so Synapse-Graph masks it "
                        "during subsequent generations."
                    ),
                    style={"color": "#39FF14"},
                )
            )
        except Exception:
            LOGGER.exception("Failed to bootstrap SynapseQuarantine governance assets in OpenMetadata.")

    def _upsert_table_sync(
        self,
        *,
        metadata: OpenMetadata,
        schema_fqn: str,
        table_name: str,
        description: str,
        columns: list[Column],
        layer_index: int | None = None,
    ) -> TableBinding:
        table_entity = metadata.create_or_update(
            data=CreateTableRequest(
                name=table_name,
                displayName=table_name,
                databaseSchema=schema_fqn,
                tableType="Regular",
                description=description,
                columns=columns,
            )
        )

        table_fqn = _fqn_value(table_entity.fullyQualifiedName)
        column_fqns = {
            _column_name(column): f"{table_fqn}.{_column_name(column)}"
            for column in columns
        }

        return TableBinding(
            table_name=table_name,
            table_fqn=table_fqn,
            table_id=_id_to_str(table_entity.id),
            layer_index=layer_index,
            column_fqns=column_fqns,
        )

    def _ingest_step_sync(
        self,
        catalog: NeuralCatalogBinding,
        session_id: str,
        prompt: str,
        step: TokenStepCapture,
    ) -> None:
        metadata = self._metadata_client()
        active_layers = [layer for layer in step.layers if layer.top_heads]
        if not active_layers:
            return

        first_binding = catalog.layer_binding(active_layers[0].layer_index)
        if first_binding is not None:
            self._add_lineage_edge_sync(
                metadata=metadata,
                source_table=catalog.prompt_table,
                target_table=first_binding,
                source_columns=[catalog.prompt_table.column_fqns["Prompt_Text"]],
                target_columns=self._resolve_active_target_columns(first_binding, active_layers[0]),
                description=f"Prompt ingress into {active_layers[0].layer_name}",
                sql_query=_build_synthetic_sql(session_id, prompt, step, active_layers[0].layer_name),
            )

        for source_layer, target_layer in zip(active_layers, active_layers[1:]):
            source_binding = catalog.layer_binding(source_layer.layer_index)
            target_binding = catalog.layer_binding(target_layer.layer_index)
            if source_binding is None or target_binding is None:
                continue
            self._add_lineage_edge_sync(
                metadata=metadata,
                source_table=source_binding,
                target_table=target_binding,
                source_columns=self._resolve_active_target_columns(source_binding, source_layer),
                target_columns=self._resolve_active_target_columns(target_binding, target_layer),
                description=f"{source_layer.layer_name} -> {target_layer.layer_name}",
                sql_query=_build_synthetic_sql(session_id, prompt, step, target_layer.layer_name),
            )

        last_layer = active_layers[-1]
        last_binding = catalog.layer_binding(last_layer.layer_index)
        if last_binding is not None:
            self._add_lineage_edge_sync(
                metadata=metadata,
                source_table=last_binding,
                target_table=catalog.response_table,
                source_columns=self._resolve_active_target_columns(last_binding, last_layer),
                target_columns=[catalog.response_table.column_fqns["Response_Text"]],
                description=f"{last_layer.layer_name} -> response egress",
                sql_query=_build_synthetic_sql(session_id, prompt, step, "Response_Egress"),
            )

    def _resolve_active_target_columns(
        self,
        table_binding: TableBinding,
        layer_capture: Any,
    ) -> list[str]:
        selected_heads = [
            head
            for head in layer_capture.top_heads
            if not head.masked and head.max_attention_score > 0
        ]
        if not selected_heads:
            selected_heads = list(layer_capture.top_heads)

        selected_columns = [
            table_binding.column_fqns[head.head_name]
            for head in selected_heads[: self.settings.openmetadata_lineage_top_heads_per_layer]
            if head.head_name in table_binding.column_fqns
        ]
        return selected_columns

    def _add_lineage_edge_sync(
        self,
        *,
        metadata: OpenMetadata,
        source_table: TableBinding,
        target_table: TableBinding,
        source_columns: list[str],
        target_columns: list[str],
        description: str,
        sql_query: str,
    ) -> None:
        if not source_columns or not target_columns:
            return

        lineage_request = AddLineageRequest(
            edge=EntitiesEdge(
                description=description,
                fromEntity=EntityReference(id=source_table.table_id, type="table"),
                toEntity=EntityReference(id=target_table.table_id, type="table"),
                lineageDetails=LineageDetails(
                    sqlQuery=sql_query,
                    columnsLineage=[
                        ColumnLineage(
                            fromColumns=source_columns,
                            toColumn=target_column,
                        )
                        for target_column in target_columns
                    ],
                ),
            )
        )

        metadata.add_lineage(data=lineage_request)

    async def _fetch_table_payload(self, table_fqn: str) -> dict[str, Any] | None:
        try:
            response = await self._request_with_managed_auth(
                "GET",
                f"/v1/tables/name/{quote(table_fqn, safe='')}",
                params={"fields": "columns,tags"},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            LOGGER.exception("Failed to fetch table payload from OpenMetadata for %s.", table_fqn)
            return None
        return response.json()

    def _resolved_auth_provider(self) -> str:
        if self.settings.openmetadata_jwt_token:
            return self.settings.openmetadata_auth_provider
        if self._uses_password_login():
            return "basic"
        # When no JWT is configured, prefer an unauthenticated client shape for
        # local OpenMetadata instances. If the server enforces auth, callers
        # must set SYNAPSE_OPENMETADATA_JWT_TOKEN.
        return "basic"

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.openmetadata_jwt_token:
            headers["Authorization"] = f"Bearer {self.settings.openmetadata_jwt_token}"
        return headers

    def _uses_password_login(self) -> bool:
        return bool(self.settings.openmetadata_email and self.settings.openmetadata_password)

    def _metadata_client(self) -> OpenMetadata:
        if self._metadata is not None:
            return self._metadata

        connection_kwargs: dict[str, Any] = {
            "hostPort": self.settings.openmetadata_host,
            "authProvider": self._resolved_auth_provider(),
        }
        if self.settings.openmetadata_jwt_token:
            connection_kwargs["securityConfig"] = OpenMetadataJWTClientConfig(
                jwtToken=self.settings.openmetadata_jwt_token
            )
        else:
            connection_kwargs["enableVersionValidation"] = False

        self._metadata = OpenMetadata(OpenMetadataConnection(**connection_kwargs))
        if self.settings.openmetadata_jwt_token:
            return self._metadata

        if self._uses_password_login():
            # The SDK client refreshes bearer tokens through this callback.
            self._metadata.client.config.auth_header = "Authorization"
            self._metadata.client.config.auth_token = self._sdk_auth_token_callback
            self._metadata.client.config.auth_token_mode = "Bearer"
            self._metadata.client.config.access_token = None
            self._metadata.client.config.expires_in = None
            bundle = self._ensure_access_token_sync()
            if bundle is not None:
                self._apply_token_bundle(bundle)
            return self._metadata

        # No JWT and no password login configured. Leave the SDK unauthenticated
        # for local no-auth OpenMetadata instances.
        self._metadata.client.config.auth_header = None
        self._metadata.client.config.auth_token = None
        self._metadata.client.config.access_token = "no_token"
        return self._metadata

    async def _request_with_managed_auth(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._uses_password_login():
            await asyncio.to_thread(self._ensure_access_token_sync)
        response = await self._client.request(method, path, **kwargs)

        if response.status_code == 401 and self._uses_password_login():
            await asyncio.to_thread(self._invalidate_access_token_sync)
            await asyncio.to_thread(self._ensure_access_token_sync)
            response = await self._client.request(method, path, **kwargs)

        return response

    def _sdk_auth_token_callback(self) -> tuple[str, datetime]:
        if self.settings.openmetadata_jwt_token:
            return (
                self.settings.openmetadata_jwt_token,
                datetime.now(timezone.utc) + timedelta(days=365),
            )

        bundle = self._ensure_access_token_sync()
        if bundle is None:
            raise RuntimeError(
                "OpenMetadata requires authentication, but no JWT or email/password "
                "credentials are configured."
            )

        expires_at = bundle.expires_at or (datetime.now(timezone.utc) + timedelta(minutes=30))
        return bundle.access_token, expires_at

    def _ensure_access_token_sync(self) -> OpenMetadataTokenBundle | None:
        if self.settings.openmetadata_jwt_token:
            return OpenMetadataTokenBundle(
                accessToken=self.settings.openmetadata_jwt_token,
                tokenType="Bearer",
            )
        if not self._uses_password_login():
            return None

        with self._token_lock:
            if self._token_bundle is not None and not self._token_is_expiring(self._token_bundle):
                return self._token_bundle

            if self._token_bundle is not None and self._token_bundle.refresh_token:
                try:
                    self._token_bundle = self._refresh_access_token_sync(self._token_bundle.refresh_token)
                    self._apply_token_bundle(self._token_bundle)
                    return self._token_bundle
                except Exception:
                    LOGGER.warning(
                        "OpenMetadata token refresh failed. Falling back to username/password login.",
                        exc_info=True,
                    )
                    self._invalidate_access_token_sync()

            self._token_bundle = self._login_sync()
            self._apply_token_bundle(self._token_bundle)
            return self._token_bundle

    def _invalidate_access_token_sync(self) -> None:
        self._token_bundle = None
        if "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]
        if self._metadata is not None:
            self._metadata.client.config.access_token = None
            self._metadata.client.config.expires_in = None

    def _token_is_expiring(self, bundle: OpenMetadataTokenBundle) -> bool:
        expires_at = bundle.expires_at
        if expires_at is None:
            return False
        refresh_at = expires_at - timedelta(seconds=self.settings.openmetadata_token_refresh_skew_seconds)
        return datetime.now(timezone.utc) >= refresh_at

    def _login_sync(self) -> OpenMetadataTokenBundle:
        password = self.settings.openmetadata_password
        email = self.settings.openmetadata_email
        if not password or not email:
            raise RuntimeError(
                "OpenMetadata email/password login is not configured. Set "
                "SYNAPSE_OPENMETADATA_EMAIL and SYNAPSE_OPENMETADATA_PASSWORD."
            )

        response = self._sync_client.post(
            "/v1/users/login",
            json={
                "email": email,
                "password": base64.b64encode(password.encode("utf-8")).decode("ascii"),
            },
        )
        response.raise_for_status()
        return OpenMetadataTokenBundle.model_validate(response.json())

    def _refresh_access_token_sync(self, refresh_token: str) -> OpenMetadataTokenBundle:
        response = self._sync_client.post(
            "/v1/users/refresh",
            json={"refreshToken": refresh_token},
        )
        response.raise_for_status()
        return OpenMetadataTokenBundle.model_validate(response.json())

    def _apply_token_bundle(self, bundle: OpenMetadataTokenBundle) -> None:
        self._client.headers["Authorization"] = bundle.authorization_header
        if self._metadata is None:
            return

        self._metadata.client.config.auth_header = "Authorization"
        self._metadata.client.config.auth_token = self._sdk_auth_token_callback
        self._metadata.client.config.auth_token_mode = bundle.token_type
        self._metadata.client.config.access_token = bundle.access_token
        self._metadata.client.config.expires_in = (
            bundle.expires_at.timestamp() if bundle.expires_at is not None else None
        )


def _sanitize_name(raw_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_")
    if not sanitized:
        return "Local_Model"
    if sanitized[0].isdigit():
        sanitized = f"Model_{sanitized}"
    return sanitized[:128]


def _normalize_openmetadata_host(raw_host: str) -> str:
    host = raw_host.rstrip("/")
    if host.endswith("/api"):
        return host
    return f"{host}/api"


def _parse_head_index(head_name: str | None) -> int | None:
    if not head_name:
        return None
    match = _HEAD_NAME_PATTERN.search(head_name)
    if not match:
        return None
    return int(match.group(1)) - 1


def _has_defective_tag(tags: list[dict[str, Any]], defective_tag_name: str) -> bool:
    normalized_target = defective_tag_name.replace("[", "").replace("]", "").upper()
    for tag_payload in tags:
        candidates = [
            tag_payload.get("tagFQN"),
            tag_payload.get("fullyQualifiedName"),
            tag_payload.get("name"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            normalized_candidate = str(candidate).split(".")[-1]
            normalized_candidate = normalized_candidate.replace("[", "").replace("]", "").upper()
            if normalized_candidate == normalized_target:
                return True
    return False


def _build_synthetic_sql(
    session_id: str,
    prompt: str,
    step: TokenStepCapture,
    target_name: str,
) -> str:
    prompt_preview = prompt.replace("\n", " ")[:160]
    activation_path = " -> ".join(step.high_activation_path[:6]) or "n/a"
    evidence_tokens = ", ".join(step.evidence_tokens[:6]) or "n/a"
    explanation = (step.explanation or "n/a").replace("\n", " ")[:220]
    return (
        f"-- Synapse-Graph synthetic lineage\n"
        f"-- session={session_id}\n"
        f"-- step={step.step_index}\n"
        f"-- target={target_name}\n"
        f"-- prompt={prompt_preview}\n"
        f"-- path={activation_path}\n"
        f"-- evidence_tokens={evidence_tokens}\n"
        f"-- explanation={explanation}\n"
        f"SELECT neural_signal FROM previous_state INTO {target_name};"
    )


def _fqn_value(value: Any) -> str:
    return str(getattr(value, "root", value))


def _column_name(column: Column) -> str:
    raw_name = getattr(column, "name", "")
    return str(getattr(raw_name, "root", raw_name))


def _id_to_str(value: Any) -> str:
    """Normalize various SDK id shapes into a plain UUID string when possible.

    The OpenMetadata SDK can return id objects that wrap a UUID (sometimes
    exposed as `.root`, sometimes as `UUID('...')` in their repr). Pydantic's
    EntityReference expects a raw UUID string (or urn:uuid:...). This helper
    attempts to extract a canonical UUID string; if not possible, it falls back
    to the best string representation.
    """
    if value is None:
        return ""

    # Unwrap common wrappers
    candidate = getattr(value, "id", None) or getattr(value, "root", value)

    # If it's already a uuid.UUID, return canonical string
    if isinstance(candidate, uuid.UUID):
        return str(candidate)

    s = str(candidate)
    # Try to extract an embedded UUID substring
    m = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", s)
    if m:
        return m.group(0)

    # As a last resort return the raw string
    return s
