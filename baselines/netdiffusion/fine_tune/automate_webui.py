#!/usr/bin/env python3

import os
import time
import unittest
import argparse
import sys
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class StableDiffusionBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1) Locate the Chrome binary you installed
        chrome_bin = os.environ.get("CHROME_BIN")
        if not chrome_bin or not os.path.isfile(chrome_bin):
            sys.exit("❌ CHROME_BIN not set or not executable. "
                     "Please export CHROME_BIN to your chrome executable.")

        # 2) Set up headless Chrome options
        chrome_options = Options()
        chrome_options.binary_location = chrome_bin
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # 3) Find chromedriver on your PATH
        driver_path = shutil.which("chromedriver")
        if not driver_path:
            sys.exit("❌ chromedriver not found on PATH. "
                     "Please add your chromedriver directory to PATH.")

        service = Service(executable_path=driver_path)
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
        cls.driver.get("http://localhost:7860")
        cls.wait = WebDriverWait(cls.driver, 30)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def navigate_to_extras_tab(self):
        """Navigate to the 'Extras' tab."""
        print("Navigating to 'Extras' tab...")
        extras_tab = self.wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Extras')]"))
        )
        extras_tab.click()
        print("✅ Clicked on 'Extras' tab.")

    def navigate_to_batch_from_directory_tab(self):
        """Navigate to the 'Batch from Directory' tab."""
        print("Navigating to 'Batch from Directory' tab...")
        batch_from_dir_tab = self.wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Batch from Directory')]"))
        )
        batch_from_dir_tab.click()
        print("✅ Clicked on 'Batch from Directory' tab.")

    def enter_directories(self):
        """Enter the input and output directories."""
        print("Entering input and output directories via CSS selectors...")
        try:
            # Input Directory
            input_dir_field = self.driver.find_element(By.CSS_SELECTOR, "#extras_batch_input_dir > label > textarea")
            input_dir_field.clear()
            input_dir_field.send_keys(self.input_dir)
            print(f"✅ Successfully entered input directory: {self.input_dir}")

            # Output Directory
            output_dir_field = self.driver.find_element(By.CSS_SELECTOR, "#extras_batch_output_dir > label > textarea")
            output_dir_field.clear()
            output_dir_field.send_keys(self.output_dir)
            print(f"✅ Successfully entered output directory: {self.output_dir}")
        except Exception as e:
            print(f"❌ Failed to enter directories: {e}")
            self.driver.save_screenshot("error_screenshot.png")
            print("📸 Screenshot saved as 'error_screenshot.png'")
            self.fail("Entering directories failed.")

    def set_scale_dimensions(self, width, height):
        """Set the width and height for 'Scale to' using JavaScript."""
        print("Setting scale dimensions...")

        try:
            # Click the "Scale to" tab using JavaScript to ensure it's interactable
            scale_tab = self.driver.find_element(By.XPATH, '//*[@id="extras_resize_mode"]/div[1]/button[2]')
            self.driver.execute_script("arguments[0].click();", scale_tab)
            print("✅ Clicked on the 'Scale to' tab using JavaScript.")

            # Use JavaScript to set the width
            width_field_js = 'document.querySelector("#extras_upscaling_resize_w > div.wrap.svelte-1cl284s > div > input").value = arguments[0]'
            self.driver.execute_script(width_field_js, str(width))
            print(f"✅ Successfully set width to {width} using JavaScript.")

            # Use JavaScript to set the height
            height_field_js = 'document.querySelector("#extras_upscaling_resize_h > div.wrap.svelte-1cl284s > div > input").value = arguments[0]'
            self.driver.execute_script(height_field_js, str(height))
            print(f"✅ Successfully set height to {height} using JavaScript.")

        except Exception as e:
            print(f"❌ Failed to set scale dimensions: {e}")
            self.driver.save_screenshot("scale_error_screenshot.png")
            print("📸 Screenshot saved as 'scale_error_screenshot.png'")
            self.fail("Setting scale dimensions failed.")

    def enable_caption_checkbox(self):
        """Enable the 'Caption' checkbox if not already selected."""
        print("Enabling 'Caption' checkbox...")
        try:
            caption_checkbox = self.driver.find_element(By.CSS_SELECTOR, "#input-accordion-6-visible-checkbox")
            if not caption_checkbox.is_selected():
                caption_checkbox.click()
                print("✅ Enabled the 'Caption' checkbox.")
            else:
                print("ℹ️ 'Caption' checkbox is already enabled.")
        except Exception as e:
            print(f"❌ Failed to enable 'Caption' checkbox: {e}")
            self.driver.save_screenshot("caption_error_screenshot.png")
            print("📸 Screenshot saved as 'caption_error_screenshot.png'")
            self.fail("Enabling caption checkbox failed.")

    def click_generate_button(self):
        """Click the 'Generate' button to start the batch process."""
        print("Clicking the 'Generate' button...")
        try:
            generate_button = self.driver.find_element(By.CSS_SELECTOR, "#extras_generate_box")
            generate_button.click()
            print("✅ Clicked the 'Generate' button.")
        except Exception as e:
            print(f"❌ Failed to click the 'Generate' button: {e}")
            self.driver.save_screenshot("generate_error_screenshot.png")
            print("📸 Screenshot saved as 'generate_error_screenshot.png'")
            self.fail("Clicking generate button failed.")

    def test_batch_from_directory(self):
        """Main test function to automate the entire process."""
        self.navigate_to_extras_tab()
        self.navigate_to_batch_from_directory_tab()
        self.enter_directories()
        self.set_scale_dimensions(816, 768)
        self.enable_caption_checkbox()
        self.click_generate_button()

        # Wait for the process to complete
        print("Waiting for the batch process to complete...")
        time.sleep(60)  # Adjust sleep time as needed
        print("✅ Batch process completed.")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Automate WebUI Interaction")
    parser.add_argument('--input_dir', type=str, required=True, help='Absolute path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Absolute path to output directory')
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args

if __name__ == "__main__":
    # Parse the arguments first
    args, remaining_args = parse_arguments()

    # Assign the parsed arguments to the test class
    StableDiffusionBatchTest.input_dir = args.input_dir
    StableDiffusionBatchTest.output_dir = args.output_dir

    # Run the tests with the remaining arguments
    unittest.main(argv=[sys.argv[0]] + remaining_args)
