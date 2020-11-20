from tests import IN_CI

try:
    import ray

    def init_ray():
        if not IN_CI:
            min_memory_possible = 52428800 * 7  # 340MB
            min_object_store_memory = 78643200 * 5  # 370Mb
            ray.init(
                _memory=min_memory_possible,
                object_store_memory=min_object_store_memory,
                ignore_reinit_error=True,
            )
        else:
            ray.init(
                local_mode=True,
                ignore_reinit_error=True,
            )


except ImportError:

    from fragile.distributed.ray import ray

    def init_ray():
        pass
