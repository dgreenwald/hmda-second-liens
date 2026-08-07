from py_tools.datasets import hmda

# hmda.download_snapshot_lar()
# hmda.download_lar(progress=True)
hmda.download_lar(source='cfpb', years=[2017], progress=True)
hmda.download_lar(source='nara', years=list(range(2007, 2015)), progress=True)
