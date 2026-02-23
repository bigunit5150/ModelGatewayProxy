"""create usage_records table

Revision ID: 001
Revises:
Create Date: 2026-02-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "usage_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "cost_usd", sa.Numeric(precision=10, scale=6), nullable=False, server_default="0"
        ),
        sa.Column("cached", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("cache_type", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_records_timestamp", "usage_records", ["timestamp"])
    op.create_index("ix_usage_records_user_id", "usage_records", ["user_id"])
    op.create_index(
        "ix_usage_records_user_id_timestamp",
        "usage_records",
        ["user_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_usage_records_user_id_timestamp", table_name="usage_records")
    op.drop_index("ix_usage_records_user_id", table_name="usage_records")
    op.drop_index("ix_usage_records_timestamp", table_name="usage_records")
    op.drop_table("usage_records")
