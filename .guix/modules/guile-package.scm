(define-module (data-science-utils)
  #:use-module (gnu)
  #:use-module (gnu packages python-xyz) ; numpy
  #:use-module (gnu packages python-build) ; python-setuptools
  #:use-module (gnu packages check) ; python-pytest
  #:use-module (guix-science packages python) ; jax
  #:use-module (guix packages) ; Record: package
  #:use-module (guix build-system pyproject) ; pyproject build system
  #:use-module (guix git-download)  ;for ‘git-predicate’
  #:use-module (guix utils)  ;for ‘current-source-directory’
  #:use-module ((guix licenses) #:prefix license:))

(define vcs-file?
  ;; Return true if the given file is under version control.
  (or (git-predicate (current-source-directory))
      (const #t)))                                ;not in a Git checkout

(package
  (name "data_science_utils")
  (version "0.99-git")
  (source (local-file "." "data_science_utils_checkout"
                      #:recursive? #t
                      #:select? vcs-file?))
  (build-system pyproject-build-system)
  (native-inputs (list python-setuptools
		       python-wheel
		       python-pytest))
  (home-page "")
  (synopsis "")
  (description "")
  (license license:agpl3+))
