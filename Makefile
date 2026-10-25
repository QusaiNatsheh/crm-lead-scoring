# Local demo with Docker. `make demo` builds everything, creates database "demo", loads the CRM history and scores.
DB ?= demo

up:
	docker compose up -d --build

demo: up
	docker compose exec odoo odoo -d $(DB) -i crm_ml_score --without-demo=all --stop-after-init
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/load_demo_data.py
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/e2e_check.py
	@echo "Open http://localhost:8071 (database $(DB), user admin / admin) -> CRM -> Reporting -> Lead Scores"

e2e:
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/e2e_check.py

reset-demo:  # fresh demo database with the CRM history loaded (run `make e2e` to score the open leads)
	docker compose exec -T -e PGPASSWORD=odoo odoo dropdb --force --if-exists -h db -U odoo $(DB)
	docker compose exec odoo odoo -d $(DB) -i crm_ml_score --without-demo=all --stop-after-init
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/load_demo_data.py
	docker compose restart odoo

screenshots:  # needs: pip install playwright pillow (uses the installed Chrome)
	python scripts/screenshots.py --url http://localhost:8071 --db $(DB)

down:
	docker compose down

.PHONY: up demo e2e reset-demo screenshots down
