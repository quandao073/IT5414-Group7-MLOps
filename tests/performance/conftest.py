"""Keep Locust from monkey-patching gevent while pytest imports the harness.

The Locust CLI still patches normally in its own process; this only affects the
test session, where patching ssl/threads after other libraries import them can
hang unrelated tests.
"""

import os

os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")
os.environ.setdefault("LOCUST_SKIP_URLLIB3_PATCH", "1")
