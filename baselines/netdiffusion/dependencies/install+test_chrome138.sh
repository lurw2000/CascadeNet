#!/usr/bin/env bash
set -euo pipefail

# 0) Where to put chrome & chromedriver
PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PREFIX"

# 1) Download Chrome for Testing v138
CHROME_VER="138.0.7204.157"
CHROME_URL="https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VER}/linux64/chrome-linux64.zip"
CHROME_ZIP="$PREFIX/chrome-${CHROME_VER}.zip"
CHROME_DIR="$PREFIX/chrome-${CHROME_VER}"

echo "→ Downloading Chrome ${CHROME_VER}..."
wget -qO "$CHROME_ZIP" "$CHROME_URL"

echo "→ Extracting Chrome to $PREFIX..."
unzip -q "$CHROME_ZIP" -d "$PREFIX"
mv "$PREFIX/chrome-linux64" "$CHROME_DIR"
chmod +x "$CHROME_DIR/chrome"

# 2) Download ChromeDriver for Testing v138
DRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VER}/linux64/chromedriver-linux64.zip"
DRIVER_ZIP="$PREFIX/chromedriver-${CHROME_VER}.zip"
DRIVER_DIR="$PREFIX/chromedriver-${CHROME_VER}"

echo "→ Downloading ChromeDriver ${CHROME_VER}..."
wget -qO "$DRIVER_ZIP" "$DRIVER_URL"

echo "→ Extracting ChromeDriver to $PREFIX..."
unzip -q "$DRIVER_ZIP" -d "$PREFIX"
mv "$PREFIX/chromedriver-linux64" "$DRIVER_DIR"
chmod +x "$DRIVER_DIR/chromedriver"

# 3) Export into this shell: Todo: still need to export from working terminal and absolute path
# ==============================================================================================
export CHROME_BIN="$CHROME_DIR/chrome"
export PATH="$DRIVER_DIR:$PATH"
# ==============================================================================================

echo "✅ CHROME_BIN set to $CHROME_BIN"
echo "✅ Added $DRIVER_DIR to PATH"

# 4) Run a quick Selenium smoke‑test
echo "→ Testing headless Chrome + ChromeDriver via Selenium…"
python3 - << 'PYCODE'
import os, sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 4a) Check that we have the browser and driver
chrome_bin = os.environ.get("CHROME_BIN")
if not chrome_bin or not os.path.isfile(chrome_bin) or not os.access(chrome_bin, os.X_OK):
    print(f"❌ CHROME_BIN not executable: {chrome_bin}")
    sys.exit(1)

from shutil import which
driver_path = which("chromedriver")
if not driver_path:
    print("❌ chromedriver not on PATH")
    sys.exit(1)

# 4b) Launch headless
opts = Options()
opts.binary_location = chrome_bin
opts.add_argument("--headless")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-gpu")
opts.add_argument("--window-size=1920,1080")

service = Service(executable_path=driver_path)
try:
    driver = webdriver.Chrome(service=service, options=opts)
    driver.get("https://www.google.com")
    print("✅ Page title is:", driver.title)
    driver.quit()
except Exception as e:
    print("❌ Selenium test failed:", e)
    sys.exit(1)
PYCODE

echo "🎉 All done—Chrome ${CHROME_VER} + ChromeDriver ${CHROME_VER} are working!"
