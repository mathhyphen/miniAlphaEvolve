"""Web UI Test Script for AlphaEvolve"""

from playwright.sync_api import sync_playwright
import sys
import subprocess
import time
import os

def test_webui():
    """Test the AlphaEvolve Web UI."""

    # Start Python HTTP server for frontend
    frontend_dir = os.path.join(os.path.dirname(__file__), "alphaevolve_webui/frontend/dist")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(2)  # Wait for server to start

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Testing AlphaEvolve Web UI...")
        print("-" * 50)

        try:
            # Navigate to the frontend
            page.goto('http://localhost:8080')
            page.wait_for_load_state('networkidle')

            print("[PASS] Frontend loaded successfully")

            # Check page title
            title = page.title()
            print(f"[INFO] Page title: {title}")

            # Check for main elements
            try:
                # Look for any visible text that contains AlphaEvolve
                content = page.content()
                if "AlphaEvolve" in content or "alphaevolve" in content.lower():
                    print("[PASS] Found AlphaEvolve branding")
                else:
                    print("[INFO] AlphaEvolve branding not found in initial load")
            except Exception as e:
                print(f"[WARN] Could not check branding: {e}")

            # Check for buttons
            try:
                buttons = page.locator('button').all()
                print(f"[INFO] Found {len(buttons)} button(s)")
                for btn in buttons[:3]:  # Show first 3
                    if btn.is_visible():
                        text = btn.text_content()
                        print(f"  - Button: {text[:30] if text else 'unnamed'}")
            except Exception as e:
                print(f"[WARN] Buttons not found: {e}")

            # Check for select dropdowns
            try:
                selects = page.locator('select').all()
                print(f"[INFO] Found {len(selects)} select dropdown(s)")
            except Exception as e:
                print(f"[WARN] Select dropdowns not found: {e}")

            # Check for any glass-card elements
            try:
                cards = page.locator('[class*="glass"]').all()
                print(f"[INFO] Found {len(cards)} glass-styled element(s)")
            except Exception as e:
                print(f"[WARN] Glass elements not found: {e}")

            # Take a screenshot
            os.makedirs("outputs", exist_ok=True)
            page.screenshot(path='outputs/webui_test_screenshot.png', full_page=True)
            print("[INFO] Screenshot saved to outputs/webui_test_screenshot.png")

            # Check for console errors
            console_errors = []
            page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' else None)
            page.reload()
            page.wait_for_load_state('networkidle')

            if console_errors:
                print(f"[WARN] Console errors found: {len(console_errors)}")
                for err in console_errors[:3]:
                    print(f"  - {err[:100]}")
            else:
                print("[PASS] No console errors detected")

            print("-" * 50)
            print("Frontend test completed!")
            return 0

        except Exception as e:
            print(f"[FAIL] Test failed: {e}")
            try:
                page.screenshot(path='outputs/webui_error_screenshot.png')
                print("[INFO] Error screenshot saved")
            except:
                pass
            return 1

        finally:
            browser.close()

    # Cleanup
    server_process.terminate()
    server_process.wait()


if __name__ == "__main__":
    sys.exit(test_webui())
