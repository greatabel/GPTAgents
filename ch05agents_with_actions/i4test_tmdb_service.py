import semantic_kernel as sk
from semantic_kernel.functions.kernel_arguments import KernelArguments
from plugins.Movies.tmdb import TMDbService

async def main():
    kernel = sk.Kernel()

    # 注册插件
    kernel.add_plugin(TMDbService(), plugin_name="TMDBService")

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_movie_genre_id",
        arguments=KernelArguments(genre_name="action")
    )
    print("Movie action genre id:", result)

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_tv_show_genre_id",
        arguments=KernelArguments(genre_name="action")
    )
    print("TV action genre id:", result)

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_top_movies_by_genre",
        arguments=KernelArguments(genre_name="action")
    )
    print("Top action movies:", result)

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_top_tv_shows_by_genre",
        arguments=KernelArguments(genre_name="action")
    )
    print("Top action TV shows:", result)

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_movie_genres",
        arguments=KernelArguments()
    )
    print("Movie genres:", result)

    result = await kernel.invoke(
        plugin_name="TMDBService",
        function_name="get_tv_show_genres",
        arguments=KernelArguments()
    )
    print("TV genres:", result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
