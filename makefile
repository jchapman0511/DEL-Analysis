.PHONY: all clean
all: Figures/active_overview Figures/tmap Data/models

Figures/active_overview:
	python $(CURDIR)/Scripts/active_overview.py
	python $(CURDIR)/Scripts/active_overlap.py
	python $(CURDIR)/Scripts/active_threshold.py
Figures/tmap:
	python $(CURDIR)/Scripts/plot_tmap.py
Data/models:
	python $(CURDIR)/Scripts/modelGen.py