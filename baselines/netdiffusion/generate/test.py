import requests
import base64
import unittest
import time
import sys
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Constants
BASE_URL = "http://127.0.0.1:7860"  # Change to your WebUI URL

class StableDiffusionControlNetTest(unittest.TestCase):
    def setUp(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.get(BASE_URL)  # Change to your actual URL
        self.wait = WebDriverWait(self.driver, 20)  # 20 seconds timeout

        # Get the image path from the environment variable
        self.image_path = os.getenv("IMAGE_PATH")
        if not self.image_path:
            raise ValueError("IMAGE_PATH environment variable is not set")

    def tearDown(self):
        self.driver.quit()

    def expand_controlnet_panel(self):
        controlnet_panel = self.driver.find_element(By.XPATH, "//*[@id='tab_txt2img']//*[@id='controlnet']")
        input_image_group = controlnet_panel.find_element(By.CSS_SELECTOR, ".cnet-input-image-group")
        if not input_image_group.is_displayed():
            controlnet_panel.click()
        self.wait.until(EC.visibility_of(input_image_group))

    def enable_controlnet_unit(self):
        controlnet_panel = self.driver.find_element(By.XPATH, "//*[@id='tab_txt2img']//*[@id='controlnet']")
        enable_checkbox = controlnet_panel.find_element(By.CSS_SELECTOR, ".cnet-unit-enabled input[type='checkbox']")
        if not enable_checkbox.is_selected():
            enable_checkbox.click()
        self.wait.until(EC.element_to_be_selected(enable_checkbox))

    def select_control_type(self, control_type: str):
        controlnet_panel = self.driver.find_element(By.XPATH, "//*[@id='tab_txt2img']//*[@id='controlnet']")
        control_type_radio = controlnet_panel.find_element(By.XPATH, f".//input[@value='{control_type}']")
        control_type_radio.click()
        time.sleep(3)  # Wait for gradio backend to update model/module

    def select_preprocessor_type(self, preprocessor_type: str):
        dropdown = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_controlnet_ControlNet-0_controlnet_preprocessor_dropdown")
        dropdown.click()
        options = dropdown.find_elements(By.XPATH, "//ul[contains(@class, 'options')]/li")
        for option in options:
            if option.text.lower() == preprocessor_type.lower():
                option.click()
                break
        time.sleep(1)  # Wait for the preprocessor to be set

    def enable_hires_fix(self):
        hires_fix_checkbox = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_hr input[type='checkbox']")
        if not hires_fix_checkbox.is_selected():
            hires_fix_checkbox.click()
            time.sleep(2)  # Wait for the UI to update

    def test_txt2img_with_controlnet(self):
        # Wait for the tabs to be available and click on 'txt2img'
        self.wait.until(EC.visibility_of_element_located((By.XPATH, "//*[@id='tabs']/*[contains(@class, 'tab-nav')]//button[text()='txt2img']")))
        self.driver.find_element(By.XPATH, "//*[@id='tabs']/*[contains(@class, 'tab-nav')]//button[text()='txt2img']").click()

        # Wait for the ControlNet panel to be available and interact with it
        self.wait.until(EC.visibility_of_element_located((By.XPATH, "//*[@id='tab_txt2img']//*[@id='controlnet']")))
        self.expand_controlnet_panel()
        self.enable_controlnet_unit()
        self.select_control_type("Canny")
        self.select_preprocessor_type("canny")

        # Wait for the model to be auto-selected
        time.sleep(5)  # Wait for 5 seconds

        # Upload the image
        controlnet_panel = self.driver.find_element(By.XPATH, "//*[@id='tab_txt2img']//*[@id='controlnet']")
        image_input = controlnet_panel.find_element(By.CSS_SELECTOR, ".cnet-input-image-group .cnet-image input[type='file']")
        image_input.send_keys(self.image_path)  # Use the current image path

        # Set prompt for network data
        prompt_input = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_prompt textarea")
        prompt_input.clear()
        prompt_input.send_keys("pixelated network data, type-0 <lora:Lora_mini:1>")

        # Set resolution
        width_input = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_width input[type='number']")
        height_input = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_height input[type='number']")
        width_input.clear()
        height_input.clear()
        width_input.send_keys(816)
        height_input.send_keys(768)

        # Set seed
        seed_input = self.driver.find_element(By.CSS_SELECTOR, "#txt2img_seed input[type='number']")
        seed_input.clear()
        seed_input.send_keys(1234)

        # Enable Hires.fix and set the resolution
        self.enable_hires_fix()

        # Generate the image
        generate_button = self.driver.find_element(By.XPATH, "//*[@id='txt2img_generate_box']")
        generate_button.click()

        # Wait for the generated image to appear
        try:
            self.wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#txt2img_results #txt2img_gallery img")))
        except:
            raise AssertionError("Generated image did not appear within the expected time.")

        # # Retrieve and save the generated image
        # generated_imgs = self.driver.find_elements(By.CSS_SELECTOR, "#txt2img_results #txt2img_gallery img")
        # for i, generated_img in enumerate(generated_imgs):
        #     # Get the image URL
        #     img_url = generated_img.get_attribute("src")
            
        #     # Check if the URL is base64 encoded and handle accordingly
        #     if img_url.startswith("data:image"):
        #         header, encoded = img_url.split(",", 1)
        #         img_content = base64.b64decode(encoded)
        #     else:
        #         img_content = requests.get(img_url).content

        #     # Save the image content to a file
        #     img_file_name = os.path.join(OUTPUT_DIR, f"{os.path.basename(self.image_path).split('.')[0]}.png")
        #     with open(img_file_name, "wb") as img_file:
        #         img_file.write(img_content)

# Utility function to run the tests with an image path
def run_test_with_image(image_path):
    suite = unittest.TestSuite()
    test_case = StableDiffusionControlNetTest('test_txt2img_with_controlnet')
    test_case._testMethodDoc = f"Test with image {image_path}"  # Add the image path to the test docstring
    suite.addTest(test_case)
    # Set the environment variable for the image path
    os.environ["IMAGE_PATH"] = image_path
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]  # Get the image path from the command-line arguments
    run_test_with_image(image_path)

