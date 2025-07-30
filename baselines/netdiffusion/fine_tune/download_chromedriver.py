#!/usr/bin/env python3

import sys
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
import os

def download_chromedriver():
    try:
        # Automatically download and install the correct ChromeDriver version
        driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
        print(f"✅ ChromeDriver downloaded successfully!")
        print(f"📁 ChromeDriver is located at: {driver_path}")
    except Exception as e:
        print(f"❌ An error occurred while downloading ChromeDriver: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_chromedriver()
