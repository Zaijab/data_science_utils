(use-modules (guix ci)
	     (ice-9 popen)
	     (guix channels))

(list
 (channel
  (name 'guix-science)
  (url "https://codeberg.org/guix-science/guix-science.git")
  (branch "master")
  (commit
   "414545462bd737959de6fa35ebd31196f60b0e59")
  (introduction
   (make-channel-introduction
    "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
    (openpgp-fingerprint
     "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
 (channel
  (name 'guix-science-nonfree)
  (url "https://codeberg.org/guix-science/guix-science-nonfree.git")
  (branch "master")
  (commit
   "4e8cb56b02df1c9839efd96e5395d860a3442ea7")
  (introduction
   (make-channel-introduction
    "58661b110325fd5d9b40e6f0177cc486a615817e"
    (openpgp-fingerprint
     "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
 (channel
  (name 'guix)
  (url "https://git.savannah.gnu.org/git/guix.git")
  (branch "master")
  (commit
   "c3e5888180b29d6efc104fab198247276723259e")
  (introduction
   (make-channel-introduction
    "9edb3f66fd807b096b48283debdcddccfea34bad"
    (openpgp-fingerprint
     "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
 #;(channel (name 'data-science-utils)
	  (url ".")
	  (branch "main")))

