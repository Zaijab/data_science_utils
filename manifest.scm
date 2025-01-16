(use-modules (gnu packages)
	     (guix profiles)
	     (guix packages)
	     (guix git-download)
	     (guix build-system pyproject)
	     ((guix licenses) #:prefix license:)
	     (gnu packages python-build)
	     (gnu packages image-processing)
	     (gnu packages python)
	     (gnu packages jupyter)
	     (gnu packages check)
	     (gnu packages python-xyz)
	     (gnu packages python-check)
	     (gnu packages pdf)
	     (gnu packages python-science)
	     (gnu packages machine-learning)
	     (guix-science packages python)
	     (guix-science packages machine-learning))

(define-public python-flax
  (package
    (name "python-flax")
    (version "0.10.2")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/google/flax")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "0dwdzp97qb1a291gnhvvs8qfqjib4q7khb3422p6sjrns9gysvbm"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:tests? #f
      #:test-flags
      '(list "--pyargs" "tests"
             ;; We don't have tensorboard
             "--ignore=tests/tensorboard_test.py"
             ;; These tests are failing bacause flax might only work
             ;; on CPUs that have AVX support.
             "--ignore=tests/serialization_test.py"
             "--ignore=tests/linen/linen_test.py"
             "--ignore=tests/linen/linen_recurrent_test.py"
             "--ignore=tests/linen/linen_dtypes_test.py"
             ;; These tests try to use a fixed number of CPUs that may
             ;; exceed the number of CPUs available at build time.
             "--ignore=tests/jax_utils_test.py")
      #:phases
      '(modify-phases %standard-phases
	 (delete 'sanity-check)
         (add-after 'unpack 'ignore-deprecations
           (lambda _
             (substitute* "pyproject.toml"
               (("\"error\",") "")))))))
    (propagated-inputs
     (list python-einops
           python-jax
           python-optax
           python-orbax-checkpoint
           python-msgpack
           python-numpy
           python-pyyaml
           python-rich
           python-tensorstore
           python-typing-extensions))
    (native-inputs
     (list opencv
           python-nbstripout
           python-ml-collections
           python-mypy
           python-pytorch
           python-pytest
           python-pytest-cov
           python-pytest-xdist
           python-setuptools-scm
           python-tensorflow))
    (home-page "https://github.com/google/flax")
    (synopsis "Neural network library for JAX designed for flexibility")
    (description "Flax is a neural network library for JAX that is
designed for flexibility.")
    (license license:asl2.0)))

(packages->manifest (list
		     python
		     jupyter
		     python-numpy
		     python-pandas
		     python-matplotlib
		     python-seaborn
		     python-scikit-learn
		     python-pytorch
		     python-tensorflow
		     python-jax
		     python-jaxtyping
		     python-beartype
		     python-flax))
