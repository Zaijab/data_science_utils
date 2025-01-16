(use-modules (guix ci)
	     (ice-9 popen)
	     (guix channels))

(list
 (channel
  (name 'guix-science)
  (url "https://codeberg.org/nihil/guix-science.git")
  (branch "master")
  (commit
   "d9870babd41998cee24d29d04cdee7f64e978b16")
  #;(introduction
   (make-channel-introduction
    "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
    (openpgp-fingerprint
     "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
 (channel
  (name 'guix)
  (url "https://git.savannah.gnu.org/git/guix.git")
  (branch "master")
  (commit
   "be5a8ee7a3ebe184a6e7bd8f7a82ea0b81046086")
  (introduction
   (make-channel-introduction
    "9edb3f66fd807b096b48283debdcddccfea34bad"
    (openpgp-fingerprint
     "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))
