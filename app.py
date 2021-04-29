import asyncio
from aiohttp import web
from typing import Any


async def handle(request):
    return web.Response(text="hello, world")


def main(args: Any = None) -> None:
    app = web.Application()
    app.add_routes([web.get('/post', handle)])
    web.run_app(app)


if __name__ == '__main__':
    main()
