.PHONY: all clean
all: Figures/active_overview Data/models
tmap: Figures/tmap

Figures/active_overview:
	python $(CURDIR)/Scripts/active_overview.py
	python $(CURDIR)/Scripts/active_overlap.py
	python $(CURDIR)/Scripts/active_thresholds.py
Figures/tmap:
	python $(CURDIR)/Scripts/plot_tmap.py
Data/models:
	python $(CURDIR)/Scripts/modelGen.py
clean:
	rm -rf Figures/*
	rm -rf Data/models