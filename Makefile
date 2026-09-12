.PHONY: up down logs test demo

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f inference-service

test:
	docker compose run --rm inference-service pytest -q

demo:
	python3 scripts/generate_traffic.py --requests 20 --fault errors
	python3 scripts/agentic_investigator.py
