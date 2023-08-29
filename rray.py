#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
# Authors: James Montgomery

"""
Ray Wrapper module to circumvent the ray package while maintaining ray-like
syntax in the code. Borrows directly from the implementation in isofit 
(see isofit/isofit/wrappers/ray.py) 

To enable, set the environment variable `GHG_DEBUG` to any value before
runtime. For example:
$ export ISOFIT_DEBUG=1
$ python parallel_mf.py ...

Additionally, you may pass it as a temporary environment variable via:
$ ISOFIT_DEBUG=1 python parallel_mf.py ...
"""
import logging
import ray


class Remote:
    def __init__(self, obj, *args, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, key):
        """
        Returns a Remote object on the key being requested. This enables
        ray.remote(Class).func.remote()
        """
        return Remote(getattr(self.obj, key))

    def remote(self, *args, **kwargs):
        return Remote(self.obj, *args, **kwargs)

    def get(self):
        return self.obj(*self.args, **self.kwargs)

    def __repr__(self):
        return f"<Remote({self.obj})>"


def __getattr__(key):
    """
    Reports any call to Ray that is not emulated
    """
    print(f"__getattr__({key})")
    logging.error(f"Unsupported operation: {key!r}")
    return lambda *a, **kw: None


def remote(obj):
    return Remote(obj)


def init(*args, **kwargs):
    logging.debug("Ray has been disabled for this run")


def get(jobs):
    if hasattr(jobs, "__iter__"):
        return [job.get() for job in jobs]
    else:
        return jobs.get()


def put(obj):
    return obj


def shutdown(*args, **kwargs):
    pass


class util:
    class ActorPool:
        def __init__(self, actors):
            """
            Emulates https://docs.ray.io/en/latest/_modules/ray/util/actor_pool.html

            Parameters
            ----------
            actors: list
                List of Remote objects to call
            """
            self.actors = [Remote(actor.get()) for actor in actors]

        def map_unordered(self, func, iterable):
            return [func(*pair).get() for pair in zip(self.actors, iterable)]
