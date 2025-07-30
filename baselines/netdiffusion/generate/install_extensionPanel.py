import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Constants
WEBUI_URL = "http://localhost:7860" 
EXTENSION_GIT_URL = "https://github.com/Mikubill/sd-webui-controlnet.git"
INSTALL_WAIT_TIME = 20  

def main():
    print("Starting the Selenium automation script for installing ControlNet extension.")

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize WebDriver
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("Chrome WebDriver initialized successfully.")
    except Exception as e:
        print(f"Error initializing Chrome WebDriver: {e}")
        return

    try:
        # Navigate to the WebUI
        driver.get(WEBUI_URL)
        print(f"Navigated to {WEBUI_URL}")
        
        wait = WebDriverWait(driver, 30)

        # Step 1: Open "Extensions" tab
        print("Navigating to the 'Extensions' tab.")
        extensions_tab_path = '//*[@id="tabs"]/div[1]/button[8]'
        extensions_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, extensions_tab_path)))
        extensions_tab.click()
        print("'Extensions' tab clicked.")

        # Step 2: Open "Install from URL" tab
        print("Navigating to the 'Install from URL' tab.")
        install_from_url_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="tabs_extensions"]/div[1]/button[3]'))
        )
        install_from_url_button.click()
        print("'Install from URL' tab clicked.")

        # Step 3: Enter the GitHub URL
        print(f"Entering the extension URL: {EXTENSION_GIT_URL}")
        url_textarea_selector = "#component-1596 > label > textarea"
        url_textarea = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, url_textarea_selector)))
        url_textarea.clear()
        url_textarea.send_keys(EXTENSION_GIT_URL)
        print("Extension URL entered.")

        # Step 4: Press "Install" button
        print("Clicking the 'Install' button.")
        install_button_selector = "#component-1599"
        install_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, install_button_selector)))
        install_button.click()
        print("'Install' button clicked.")

        # Step 5: Wait for installation message
        print("Waiting for installation to complete.")
        installation_message = "Installed into stable-diffusion-webui\\extensions\\sd-webui-controlnet. Use Installed tab to restart"
        try:
            wait.until(EC.text_to_be_present_in_element(
                (By.TAG_NAME, "body"), installation_message))
            print("Installation completed successfully.")
        except:
            print("Warning: Installation message not found within the wait time.")

        # Step 6: Go to "Installed" tab, check for updates, and apply
        print("Navigating to the 'Installed' tab.")
        installed_tab_selector = "#tabs_extensions > div.tab-nav.scroll-hide.svelte-kqij2n > button"
        installed_tab = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, installed_tab_selector)))
        installed_tab.click()
        print("'Installed' tab clicked.")

        # Click "Check for updates"
        print("Clicking 'Check for updates' button.")
        check_updates_selector = "#component-1566"
        check_updates_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, check_updates_selector)))
        check_updates_button.click()
        print("'Check for updates' button clicked.")

        # Click "Apply and restart UI"
        print("Clicking 'Apply and restart UI' button.")
        apply_restart_selector = "#component-1565"
        apply_restart_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, apply_restart_selector)))
        apply_restart_button.click()
        print("'Apply and restart UI' button clicked.")

        print("Automation steps 1 through 6 completed successfully.")

    except Exception as e:
        print(f"An error occurred during automation: {e}")
    finally:
        # Optionally, close the browser after a delay
        print("Closing the browser in 10 seconds.")
        time.sleep(10)
        driver.quit()
        print("Browser closed. Please proceed with step 7 manually.")

if __name__ == "__main__":
    main()
