class CommonDay(object):
	"""docstring for Day"""
	def __init__(self):
		super(CommonDay, self).__init__()
		self.name = "Common Day"
		self.data = None

	def readline(self):
		pass

	def first_solver(self):
		pass

	def second_solver(self):
		pass

	def __call__(self):
		return self.first_solver(), self.second_solver(), self.name

		