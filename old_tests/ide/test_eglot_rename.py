class A(object):

    def important(self):
        return 1


class B(A):

    def important(self):
        return 2


my_B = B()
my_B.important()
