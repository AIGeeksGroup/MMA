from typing import TYPE_CHECKING, Optional
from datetime import datetime
import datetime as dt

from sqlalchemy import Column, DateTime, String, JSON, Index, text, Float
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship

from mma.orm.sqlalchemy_base import SqlalchemyBase
from mma.orm.mixins import OrganizationMixin

from mma.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mma.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mma.constants import MAX_EMBEDDING_DIM
from mma.settings import settings

if TYPE_CHECKING:
    from mma.orm.organization import Organization


class ProceduralMemoryItem(SqlalchemyBase, OrganizationMixin):
    """
    Stores procedural memory entries, such as workflows, step-by-step guides, or how-to knowledge.
    
    type:        The category or tag of the procedure (e.g. 'workflow', 'guide', 'script')
    description: Short descriptive text about what this procedure accomplishes
    steps:       Step-by-step instructions or method
    metadata_:   Additional fields/notes
    """

    __tablename__ = "procedural_memory"
    __pydantic_model__ = PydanticProceduralMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this procedural memory entry",
    )

    # Distinguish the type/category of the procedure
    entry_type: Mapped[str] = mapped_column(
        String,
        doc="Category or type (e.g. 'workflow', 'guide', 'script')"
    )

    # A human-friendly description of this procedure
    summary: Mapped[str] = mapped_column(
        String,
        doc="Short description or title of the procedure"
    )

    # Steps or instructions stored as a JSON object/list
    steps: Mapped[list] = mapped_column(
        JSON,
        doc="Step-by-step instructions stored as a list of strings"
    )

    # Hierarchical categorization path
    tree_path: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="Hierarchical categorization path as an array of strings"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {"timestamp": datetime.now(dt.timezone.utc).isoformat(), "operation": "created"},
        doc="Last modification info including timestamp and operation type"
    )

    # Optional metadata
    metadata_: Mapped[dict] = mapped_column(
        JSON,
        default={},
        nullable=True,
        doc="Arbitrary additional metadata as a JSON object"
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, 
        nullable=True,
        doc="Embedding configuration"
    )

    source_tag: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Confidence source tag for V1 confidence scoring"
    )
    
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        doc="Persistent intrinsic reliability (V2)"
    )
    
    links: Mapped[Optional[dict]] = mapped_column(
        JSON, 
        default=dict, 
        nullable=True, 
        doc="Persistent semantic links between memories"
    )
    
    # Vector embedding field based on database type
    if settings.mma_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector
        summary_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        steps_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        summary_embedding = Column(CommonVector, nullable=True)
        steps_embedding = Column(CommonVector, nullable=True)


    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        Adjust 'back_populates' to match the collection name in your `Organization` model.
        """
        return relationship(
            "Organization",
            back_populates="procedural_memory",
            lazy="selectin"
        )
