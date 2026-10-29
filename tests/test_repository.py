"""Repository writes leave transaction decisions to their caller."""

from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import BaseDBModel
from app.helpers.repository import BaseRepository


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["save", "delete"])
async def test_writes_leave_transaction_open(operation):
    repository = BaseRepository(BaseDBModel)
    session = AsyncMock(spec=AsyncSession)
    instance = Mock(spec=BaseDBModel)

    result = await getattr(repository, operation)(session, instance)

    assert result is (instance if operation == "save" else None)
    session.flush.assert_awaited_once()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()
    session.refresh.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "operation", ["save", "delete", "save_and_commit", "delete_and_commit"]
)
async def test_flush_failure_propagates_to_transaction_owner(operation):
    repository = BaseRepository(BaseDBModel)
    session = AsyncMock(spec=AsyncSession)
    session.flush.side_effect = RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        await getattr(repository, operation)(session, Mock(spec=BaseDBModel))

    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["save_and_commit", "delete_and_commit"])
async def test_commit_helpers_flush_before_committing(operation):
    repository = BaseRepository(BaseDBModel)
    session = AsyncMock(spec=AsyncSession)
    instance = Mock(spec=BaseDBModel)

    async def verify_flushed():
        session.flush.assert_awaited_once()

    session.commit.side_effect = verify_flushed
    result = await getattr(repository, operation)(session, instance)

    assert result is (instance if operation == "save_and_commit" else None)
    session.commit.assert_awaited_once()
    if operation == "save_and_commit":
        session.add.assert_called_once_with(instance)
    else:
        session.delete.assert_awaited_once_with(instance)
