from datetime import datetime
from typing import Optional

from pydantic import Field

from mma.schemas.mma_base import MMABase
from mma.utils import create_random_username, get_utc_time


class OrganizationBase(MMABase):
    __id_prefix__ = "org"


class Organization(OrganizationBase):
    id: str = OrganizationBase.generate_id_field()
    name: str = Field(create_random_username(), description="The name of the organization.", json_schema_extra={"default": "SincereYogurt"})
    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the organization.")


class OrganizationCreate(OrganizationBase):
    name: Optional[str] = Field(None, description="The name of the organization.")
