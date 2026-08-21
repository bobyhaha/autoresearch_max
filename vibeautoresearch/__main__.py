import json
import sys

from .cli import main
from .core import SchemaError


try:
    raise SystemExit(main())
except SchemaError as exc:
    print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
    raise SystemExit(2) from None
