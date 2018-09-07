if __name__ == '__main__':
    get_ipython().run_line_magic('pylab', 'inline')

import glob, os, pandas, pickle, fnmatch, numpy, scipy.stats, math
from itertools import compress
import zipfile, zlib ## for compression/uncompression
from enum import Enum

class CSV(Enum):
    LOD=0
    MAX=1
    POP=2
def _convertCSVEnumToStringFilename(file):
    if file == CSV.LOD: return 'LOD_data.csv'
    elif file == CSV.MAX: return 'max.csv'
    elif file == CSV.POP: return 'pop.csv'
    else: raise TypeError('csv Enum file type {filetype} not recognized'.format(file))

def conditions(path=''):
    '''path: path to dir containing all conditions
    yields: each condition dir name found'''
    condDirs = glob.glob(os.path.join(path,'C*/'))
    for eachDir in condDirs:
        yield eachDir
def listConditions(path=''):
    '''path: path to dir containing all conditions
    prints: all condition names found (dir names)'''
    for eachCondition in conditions(path):
        print(eachCondition)

def __getSupersetAndReplicateSetsByCondition(path=''):
    '''path: path to dir containing all conditions
    returns: (superset, dict[condition,set])
    where superset is union of all replicates across all conditions
    where dict is set of reps indexed by condition name'''
    repSetsByCondition=dict()
    for eachCondition in conditions(path):
        subdirs = next(os.walk(eachCondition))[1]
        repSetsByCondition[eachCondition] = set([int(e) for e in subdirs])
    superset = set()
    for eachCondition,eachSet in repSetsByCondition.items():
        superset |= eachSet
    return superset, repSetsByCondition
def replicates(path=''):
    '''path: path to dir containing all conditions
    yields: each replicate from superset across all conditions'''
    superset, _ = __getSupersetAndReplicateSetsByCondition(path)
    for eachRep in list(superset):
        yield eachRep
def listReplicates(path=''):
    '''path: path to dir containing all conditions
    prints: superset of all replicates across all conditions'''
    print(list(replicates()))

def missingReplicates(path=''):
    '''path: path to dir containing all conditions
    yields: conditionNameWithMissing, repsMissing'''
    superset, repSetsByCondition = __getReplicateSetsByCondition(path)
    missingRepSetsByCondition = {}
    for eachCondition, eachRepSet in repSetsByCondition.items():
        missingRepSetsByCondition[eachCondition] = superset-eachRepSet
    for eachCondition, eachRepSet in missingRepSetsByCondition.items():
        if len(eachRepSet) > 0:
            yield eachCondition, list(eachRepSet)
def listMissingReplicates(path=''):
    '''path: path to dir containing all conditions
    prints: conditionNameWithMissing, repsMissing'''
    for eachCondition, eachRepSet in missingReplicates(path):
        print(eachCondition, eachRepSet)

def csvFiles(path=''):
    '''path: path to dir containing all conditions
    yields: filenames of all unique csv files found among all conditions and replicates'''
    names = set()
    for eachCondition in conditions(path):
        reps = next(os.walk(eachCondition))[1]
        for eachRep in reps:
            files = next(os.walk(os.path.join(eachCondition,eachRep)))[2]
            for eachFile in files:
                if eachFile.endswith('.csv'):
                    names.add(eachFile)
    for eachName in names:
        yield eachName
def listCsvFiles(path=''):
    '''path: path to dir containing all conditions
    prints: filenames of all unique csv files found among all conditions and replicates'''
    for name in csvFiles():
        print(name)

def columns(path='',condition='',file=''):
    if isinstance(file,CSV): file = _convertCSVEnumToStringFilename(file)
    if len(file) == 0: raise ValueError('file argument necessary')
    if len(condition) == 0: raise ValueError('condition argument necessary')
    firstRep = str(list(replicates(path))[0])
    csvFilePath = os.path.join(path,condition,firstRep,file)
    firstRow = pandas.read_csv(csvFilePath,nrows=1)
    colnames = firstRow.columns.values
    for eachName in colnames:
        yield eachName
def listColumns(path='',condition='',file=''):
    print('\n'.join(list(columns(path,condition,file))))

def cacheCsvOfCondition(condition,file,path='',columns=None,skiprows=None):
    '''path: path to dir containing all conditions
    condition: condition name to cache
    creates hidden cache files of conglomerate data for later use'''
    csvFileList = list(csvFiles(path))
    masterDataByCsv = {}
    for eachRep in next(os.walk(os.path.join(path,condition)))[1]:
        tempfile = pandas.read_csv(os.path.join(path,condition,str(eachRep),file),nrows=1)
        data = pandas.read_csv(os.path.join(path,condition,str(eachRep),file),usecols=columns,skiprows=skiprows)
        data['repID'] = int(eachRep) ## set new column of repid
        data['conditionID'] = condition
        data=data.rename(columns = {'score_AVE':'score'})
        if file not in masterDataByCsv: masterDataByCsv[file] = [data]
        else: masterDataByCsv[file].append(data)
    for eachCsvFilename,eachData in masterDataByCsv.items():
        pickle.dump(pandas.concat(eachData),open(os.path.join(path,condition,'.'+eachCsvFilename),'wb'))

def cacheCsvsOfAllConditions(path=''): ##TODO FIX THIS add params to cacheCsvOfCondition call
    for eachCondition in conditions(path):
        print('caching '+eachCondition+'...',end='')
        cacheCsvOfCondition(eachCondition)
        print('done')

def CITuple(data):
    ## WARNING: for 1 datum on an x-axis only (calcs the CI for one x-point)
    ## Bootstrapping 95% confidence intervals
    ## 100 samples are taken, each one is a "sample with replacement"
    ## of the original data, as points are there are originally in the data,
    ## and calculates a mean. So 100 means yields a distribution, and we take the
    ## tails of that distribution to specify the 95% CI interval.
    Nsamples=100
    means = numpy.zeros(Nsamples)
    for i in range(Nsamples):
        means[i] = mean(numpy.random.choice(data,size=len(data),replace=True))
    means = sort(means)
    lowerCIM = means[int(0.0455*Nsamples)]
    upperCIM = means[int(0.9545*Nsamples)]
    return([lowerCIM,upperCIM])

def CIError(data):
    ## WARNING: for 1 datum on an x-axis only (calcs the CI for one x-point)
    ## Bootstrapping 95% confidence intervals
    ## 100 samples are taken, each one is a "sample with replacement"
    ## of the original data, as points are there are originally in the data,
    ## and calculates a mean. So 100 means yields a distribution, and we take the
    ## tails of that distribution to specify the 95% CI interval.
    Nsamples=100
    sample = numpy.random.choice(data,size=Nsamples,replace=True)
    sample_mean = sample.mean()
    t_critical = scipy.stats.t.ppf(q = 0.975, df=Nsamples-1)
    sample_stdev = sample.std()
    sigma = sample_stdev/math.sqrt(Nsamples)
    conf_error = t_critical * sigma
    return conf_error

class Data(object):
    def __init__(self, data):
        self._data = data
    def subsetByCondition(self,substr):
        return Data(self._data[self._data['conditionID'].str.contains(substr)])
    def showEvolutionOf(self,column):
        pt = self._data.pivot_table(values=column, index='update', columns='conditionID', aggfunc=[numpy.mean,CIError])
        for eachCondition in list(compress(pt['mean'].columns.values,[True,True,True])):
            fill_between(pt['mean'][eachCondition].index,pt['mean'][eachCondition]-pt['CIError'][eachCondition],pt['mean'][eachCondition]+pt['CIError'][eachCondition],alpha=0.3)
            plot(pt['mean'][eachCondition])
    def showEvolvedOf(self,column):
        pass##pt = self._data.iloc[0].pivot_table()
class Loader(object):
    def __init__(self, path=''):
        self._path = ''
        self._conditions = None
        self._file = None
        self._data = None
        self._columns = None
        self._skiprows = None
    def useFile(self, file): ## singular
        if not isinstance(file, CSV) and not isinstance(file,str):
            raise TypeError('file must be an instance of CSV Enum or a string (full filename of csv file)')
        if isinstance(file, CSV): file = _convertCSVEnumToStringFilename(file)
        if self._file is not None: raise TypeError('Do not define multiple csv files to load from in one invocation')
        self._file = file
        return self
    def getCondition(self,pattern): ## can chain multiples of this
        if self._conditions is None: self._conditions = []
        for eachCondition in conditions():
            if fnmatch.fnmatch(eachCondition,pattern): self._conditions.append(eachCondition)
        return self
    def getColumn(self,name):
        if self._columns is None: self._columns = set() ##set(['update']) ##default columns here
        self._columns.add(name)
        return self
    def sampleEvery(self,N):
        self._skiprows = (lambda x: ((x%N)!=0)) ## mask for which rows to skip (True = skip)
        return self
    def load(self,recache=False):
        if self._conditions is None: self._conditions = list(conditions(self._path))
        if self._file is None: raise TypeError('Please specify a csv file')
        for eachCondition in self._conditions:
            cacheFilePath = os.path.join(self._path,eachCondition,'.'+self._file)
            if not os.path.exists(cacheFilePath) or recache: ## cache if not already cached
                fileCols = list(columns(self._path,eachCondition,self._file))
                filterCols = set(list(self._columns)[:])
                if filterCols is not None: ## This section fixes 'score' if only 'score_AVE' is available, for example.
                    for eachColumn in filterCols:
                        if eachColumn not in fileCols:
                            if eachColumn+'_AVE' not in fileCols:
                                raise ValueError('neither '+eachColumn+' nor '+eachColumn+'_AVE'+' appears in '+eachCondition+self._file)
                            else:
                                filterCols.remove(eachColumn)
                                filterCols.add(eachColumn+'_AVE')
                cacheCsvOfCondition(eachCondition,self._file,path=self._path,columns=filterCols,skiprows=self._skiprows)
        allDataFrames = [pickle.load(open(os.path.join(self._path,eachCondition,'.'+self._file),'rb')) for eachCondition in self._conditions]
        self._data = pandas.concat(allDataFrames)
        return self
    def data(self):
        return Data(self._data)
    def compress(self,filename):
        if self._conditions is None: self._conditions = list(conditions(self._path))
        if self._file is None: self._file = 'pop.csv'
        zf = zipfile.ZipFile(filename,mode='w')
        try:
            for eachCondition in self._conditions:
                candidateFile = os.path.join(self._path,eachCondition,'.'+self._file)
                if not os.path.isfile(candidateFile): continue
                zf.write(candidateFile, compress_type = zipfile.ZIP_DEFLATED)
        finally:
            zf.close()

if __name__ == '__main__':
    data = Loader().useFile(CSV.POP).getCondition('*DWP_1*').getColumn('update').getColumn('score').sampleEvery(100).load(recache=True).data()

if __name__ == '__main__':
    data = Loader().useFile(CSV.LOD).getColumn('score').load(recache=True).data()
    print(data._data.head())

if __name__ == '__main__':
    d = data.subsetByCondition('PathAssociation')._data
    MRCA = d['update'].unique()[-10]
    d = d[d['update'] == MRCA]
    pt = d.pivot_table(values='score_AVE', index='update', columns='conditionID', aggfunc=[numpy.mean,CIError])
    pt['mean'].plot(kind='bar')

## Uses CIError
if __name__ == '__main__':
    data = Loader().useFile(CSV.LOD).getColumn('update').getColumn('score').load().data()
    d = data.subsetByCondition('PathAssociation')._data#.showEvolutionOf('score_AVE')
    pt = d.pivot_table(values='score_AVE', index='update', columns='conditionID', aggfunc=[numpy.mean,CIError])
    for eachCondition in list(compress(pt['mean'].columns.values,[True,True,True])):
        fill_between(pt['mean'][eachCondition].index,pt['mean'][eachCondition]-pt['CIError'][eachCondition],pt['mean'][eachCondition]+pt['CIError'][eachCondition],alpha=0.3)
        plot(pt['mean'][eachCondition])
    gca().set_yscale('log')
    gca().set_xscale('log')
    ylim(bottom=1000)


# In[125]:


## Uses CITuple
if __name__ == '__main__':
    data = Loader().useFile(CSV.LOD).getColumn('update').getColumn('score').load().data()
    d = data.subsetByCondition('PathAssociation')._data#.showEvolutionOf('score_AVE')
    pt = d.pivot_table(values='score_AVE', index='update', columns='conditionID', aggfunc=[numpy.mean,CIError])
    for eachCondition in pt['mean'].columns.values:
        fill_between(pt['mean'][eachCondition].index,[e[0] for e in pt['CITuple'][eachCondition]],[e[1] for e in pt['CIError'][eachCondition]],alpha=0.3)
        plot(pt['mean'][eachCondition])


# In[136]:


if __name__ == '__main__':
    
    #for eachCondition in pt['mean'].columns.values[:
    #    fill_between(pt['mean'][eachCondition].index,[e[0] for e in pt['CIError'][eachCondition]],[e[1] for e in pt['CIError'][eachCondition]],alpha=0.3)
    #    plot(pt['mean'][eachCondition])
    ylim(bottom=3180,top=3225)
    #legend(['DEF','DWD','DWP'])


# In[91]:


if __name__ == '__main__':
    print(pt.columns.values)
    fill_between(pt['mean']['C03__DEF_1__DWD_0__DWP_0__WORLD_PathAssociation/'].index,pt['mean']['C03__DEF_1__DWD_0__DWP_0__WORLD_PathAssociation/'])
    #plot(pt['mean']['C03__DEF_1__DWD_0__DWP_0__WORLD_PathAssociation/'].index,pt['mean']['C03__DEF_1__DWD_0__DWP_0__WORLD_PathAssociation/'])


# In[69]:


if __name__ == '__main__':
    data = Loader().useFile(CSV.LOD).getColumn('update').getColumn('score').load().data()
    for eachCondition in ['Memory','Berry','EdlundMaze','PathAssociation','ValueJudgment']:
        data.subsetByCondition(eachCondition).showEvolutionOf('score_AVE')
        legend(['DEF','DWD','DWP'])

