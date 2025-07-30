from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType  # Adjust based on your setup

def test_chromedriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--verbose")
    chrome_options.binary_location = "/opt/google/chrome/chrome"

    try:
        service = ChromeService(
            ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install(),
            log_path='chromedriver_test.log'
        )
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get("https://www.google.com")
        print("Page title is:", driver.title)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_chromedriver()
