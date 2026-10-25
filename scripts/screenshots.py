"""Take README screenshots and a demo GIF from the running Docker demo (`make demo` first).

Needs Playwright with a Chrome/Chromium install:  pip install playwright pillow
Usage: python scripts/screenshots.py [--url http://localhost:8071] [--out docs/screenshots]
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1440, "height": 900}
HIDE_TOURS = (
    "document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style');"
    " s.textContent = '.o_tour_pointer { display: none !important; }'; document.head.appendChild(s); });"
)
RPC = """async ([model, method, args, kwargs]) => {
    const r = await fetch('/web/dataset/call_kw', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({jsonrpc: '2.0', method: 'call', params: {model, method, args, kwargs}})});
    return (await r.json()).result;
}"""


def settle(page) -> None:
    """Odoo keeps a live connection open (bus), so wait for the rendered view instead of network idle."""
    page.wait_for_selector(".o_action_manager .o_view_controller", timeout=30000)
    page.wait_for_timeout(1500)
    page.mouse.move(5, 890)
    page.evaluate("document.activeElement && document.activeElement.blur()")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8071")
    parser.add_argument("--db", default="demo")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []

    def shot(page, name: str, frame: bool = True) -> None:
        png = page.screenshot()
        (args.out / f"{name}.png").write_bytes(png)
        if frame:
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.add_init_script(HIDE_TOURS)
        page.goto(f"{args.url}/web/login?db={args.db}")
        page.fill("input[name=login]", "admin")
        page.fill("input[name=password]", "admin")
        page.click("button[type=submit]")
        page.wait_for_selector(".o_main_navbar", timeout=30000)

        page.goto(f"{args.url}/odoo/action-crm.crm_lead_action_pipeline")
        settle(page)
        shot(page, "01_pipeline")

        # A scored opportunity in Negotiation with a high score, to show the reasons tab
        ids = page.evaluate(
            RPC,
            [
                "crm.lead",
                "search",
                [[["ml_band", "=", "hot"], ["name", "=like", "% (#%)"]]],
                {"limit": 1, "order": "ml_probability desc"},
            ],
        )
        page.goto(f"{args.url}/odoo/action-crm.crm_lead_action_pipeline/{ids[0]}")
        page.wait_for_selector(".o_form_view")
        page.locator("a[name=ml_score], .nav-link:has-text('ML Score')").first.click()
        page.wait_for_timeout(1200)
        page.mouse.move(5, 890)
        shot(page, "02_lead_ml_tab")

        page.goto(f"{args.url}/odoo/action-crm_ml_score.action_crm_ml_score")
        settle(page)
        shot(page, "03_lead_scores")

        page.locator(".o_switch_view.o_graph").click()
        settle(page)
        shot(page, "04_pipeline_by_band", frame=False)
        browser.close()

    gif = args.out / "demo.gif"
    small = [f.resize((960, 600), Image.Resampling.LANCZOS) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=[3000, 4000, 3500], loop=0, optimize=True)
    print(f"{len(frames)} frames -> {gif}; screenshots in {args.out}")


if __name__ == "__main__":
    main()
