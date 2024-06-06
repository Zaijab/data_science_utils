(use-modules (gnu packages)
	     (guix profiles)
	     (guix-science packages python))

(packages->manifest
 (list
  (specification->package "python-numpy")
  (specification->package "python-pandas")
  (specification->package "python-matplotlib")
  (specification->package "python-seaborn")
  (specification->package "python-scikit-learn")
  (specification->package "python-pytorch")
  python-tensorflow))
