"""initial_schema

Revision ID: 753137e5e609
Revises: 
Create Date: 2026-11-15 19:49:37.387910

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '753137e5e609'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create analysis_jobs table
    op.create_table(
        'analysis_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('total', sa.Integer(), nullable=False),
        sa.Column('completed', sa.Integer(), nullable=False),
        sa.Column('failed', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_jobs_status'), 'analysis_jobs', ['status'], unique=False)

    # 2. Create reviews table
    op.create_table(
        'reviews',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('overall_label', sa.String(length=20), nullable=False),
        sa.Column('overall_score', sa.Integer(), nullable=False),
        sa.Column('theme', sa.String(length=50), nullable=False),
        sa.Column('urgency', sa.String(length=20), nullable=False),
        sa.Column('root_cause', sa.Text(), nullable=True),
        sa.Column('suggested_action', sa.Text(), nullable=True),
        sa.Column('source', sa.String(length=20), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['analysis_jobs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_reviews_created_at'), 'reviews', ['created_at'], unique=False)
    op.create_index(op.f('ix_reviews_job_id'), 'reviews', ['job_id'], unique=False)
    op.create_index(op.f('ix_reviews_overall_label'), 'reviews', ['overall_label'], unique=False)
    op.create_index(op.f('ix_reviews_overall_score'), 'reviews', ['overall_score'], unique=False)
    op.create_index(op.f('ix_reviews_theme'), 'reviews', ['theme'], unique=False)
    op.create_index(op.f('ix_reviews_urgency'), 'reviews', ['urgency'], unique=False)

    # 3. Create review_aspects table
    op.create_table(
        'review_aspects',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('review_id', sa.String(length=36), nullable=False),
        sa.Column('aspect', sa.String(length=50), nullable=False),
        sa.Column('sentiment', sa.String(length=20), nullable=False),
        sa.Column('score', sa.Integer(), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['review_id'], ['reviews.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_review_aspects_aspect'), 'review_aspects', ['aspect'], unique=False)
    op.create_index(op.f('ix_review_aspects_review_id'), 'review_aspects', ['review_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_review_aspects_review_id'), table_name='review_aspects')
    op.drop_index(op.f('ix_review_aspects_aspect'), table_name='review_aspects')
    op.drop_table('review_aspects')

    op.drop_index(op.f('ix_reviews_urgency'), table_name='reviews')
    op.drop_index(op.f('ix_reviews_theme'), table_name='reviews')
    op.drop_index(op.f('ix_reviews_overall_score'), table_name='reviews')
    op.drop_index(op.f('ix_reviews_overall_label'), table_name='reviews')
    op.drop_index(op.f('ix_reviews_job_id'), table_name='reviews')
    op.drop_index(op.f('ix_reviews_created_at'), table_name='reviews')
    op.drop_table('reviews')

    op.drop_index(op.f('ix_analysis_jobs_status'), table_name='analysis_jobs')
    op.drop_table('analysis_jobs')
