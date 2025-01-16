all: gnew

gnew:
	git add -A
	git diff-index --quiet HEAD || git commit -am "Updating Utils"
	git push -u github main
