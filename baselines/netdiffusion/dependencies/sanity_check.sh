export PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$PREFIX/bin/nprint" ]; then
  echo "nprint binary not found in $PREFIX/bin"
  exit 1
fi

export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

if nprint --help 2>&1 | head -n1 | grep -q '^Usage:'; then
  echo "nprint is on your PATH and runnable"
else
  echo "nprint ran into an error"
  exit 1
fi