from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import BaseDBModel


class BaseRepository[Model: BaseDBModel]:
    def __init__(self, model: type[Model]) -> None:
        self.model = model

    async def get(self, session: AsyncSession, id: int) -> Model | None:
        return await session.get(self.model, id)

    async def save(self, session: AsyncSession, instance: Model) -> Model:
        """Add or update an instance and flush without committing."""
        session.add(instance)
        await session.flush()
        return instance

    async def save_and_commit(self, session: AsyncSession, instance: Model) -> Model:
        """Save an instance and commit all pending changes in the session."""
        await self.save(session, instance)
        await session.commit()
        return instance

    async def delete(self, session: AsyncSession, instance: Model) -> None:
        """Delete an instance and flush without committing."""
        await session.delete(instance)
        await session.flush()

    async def delete_and_commit(self, session: AsyncSession, instance: Model) -> None:
        """Delete an instance and commit all pending changes in the session."""
        await self.delete(session, instance)
        await session.commit()
