
import asyncio
from threading import Lock



class SemaphoreLock:
	def __init__(self, max_entries, sleep_interval=0.1):
		self.max_entries = max_entries
		self.sleep_interval = sleep_interval
		self.current_entries = 0
		self.lock = Lock()

	async def __aenter__(self):
		while True:
			with self.lock:
				if self.current_entries < self.max_entries:
					self.current_entries += 1
					return
			await asyncio.sleep(self.sleep_interval)  # Wait for 100ms before trying again

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		with self.lock:
			self.current_entries -= 1


class NSlotLock:
	def __init__ (self, table, n, sleep_interval=0.1):
		self.table = table
		self.n = n
		self.key = None
		self.sleep_interval = sleep_interval
		self.lock = Lock()

	async def __aenter__ (self):
		#print('__enter__')
		while True:
			with self.lock:
				self.key = self.get_spare_key()
				#print(f'{self.key=}')
				if self.key is not None:
					break
			await asyncio.sleep(self.sleep_interval)

		return self

	async def __aexit__ (self, exc_type, exc_val, exc_tb):
		with self.lock:
			if self.key in self.table:
				del self.table[self.key]
		#print(f'__aexit__: {self.table=}')


	def get_spare_key(self):
		for i in range(self.n):
			#print(f'{i=}')
			#print(f'{self.table=}')
			if i not in self.table:
				self.table[i] = True
				return i
		return None
