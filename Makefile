.PHONY: reset
reset:
	rm -rf tmp/

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
