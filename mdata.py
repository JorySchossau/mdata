if __name__ == '__main__':
    get_ipython().magic('pylab inline')

import glob, os, pandas, pickle, fnmatch, numpy, matplotlib.pyplot as plt
import zipfile, zlib ## for compression/uncompression
from enum import Enum

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
        means[i] = numpy.mean(numpy.random.choice(data,size=len(data),replace=True))
    means = numpy.sort(means)
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
    superset, repSetsByCondition = __getSupersetAndReplicateSetsByCondition(path)
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
def listCSVFiles(path=''):
    '''path: path to dir containing all conditions
    prints: filenames of all unique csv files found among all conditions and replicates'''
    for name in csvFiles():
        print(name)

def columns(path='',condition='',file=''):
    if isinstance(file,CSV): file = _convertCSVEnumToStringFilename(file)
    if len(file) == 0: raise ValueError('file argument necessary')
    if len(condition) == 0: raise ValueError('condition argument necessary')
    firstRep = str(list(replicates(path))[0])
    fullpathFile = os.path.join(path,condition,firstRep,file)
    fullpathCacheOfFile = os.path.join(path,condition,'.'+file)
    if os.path.exists(fullpathFile):
        csvFilePath = os.path.join(path,condition,firstRep,file)
        firstRow = pandas.read_csv(csvFilePath,nrows=1)
        colnames = firstRow.columns.values ## final important variable
    elif os.path.exists(fullpathCacheOfFile):
        cachedata = pickle.load(open(fullpathCacheOfFile,'rb'))
        colnames = list(cachedata.columns) ## final important variable
    if not os.path.exists(fullpathFile) and not os.path.exists(fullpathCacheOfFile):
        raise ValueError('csv file {} doesn\'t exist and there\'s no cache.'.format(file))
    for eachName in colnames:
        yield eachName
def listColumns(path='',condition='',file=''):
    print('\n'.join(list(columns(path,condition,file))))

def __getSupersetAndColumnSetsByCondition(path='',file=''):
    '''path: path to dir containing all conditions
    returns: (superset, dict[condition,set])
    where superset is union of all column names across all conditions
    where dict is set of column names indexed by condition name'''
    colSetsByCondition=dict()
    for eachCondition in conditions(path):
        subdirs = next(os.walk(eachCondition))[1]
        colSetsByCondition[eachCondition] = set(columns(path,eachCondition,file))
    superset = set()
    for eachCondition,eachSet in colSetsByCondition.items():
        superset |= eachSet
    return superset, colSetsByCondition

def cacheCsvOfCondition(condition,file,path='',columns=None,skiprows=None,zeroColumns=None):
    '''path: path to dir containing all conditions
    condition: condition name to cache
    creates hidden cache files of conglomerate data for later use'''
    csvFileList = list(csvFiles(path))
    masterDataByCsv = {}
    print('caching '+condition+'[',end='')
    if skiprows is None:
        rowskiplambda = (lambda x: False)
    else:
        rowskiplambda = (lambda x: ((x%skiprows)!=0))
    for eachRep in next(os.walk(os.path.join(path,condition)))[1]:
        tempfile = pandas.read_csv(os.path.join(path,condition,str(eachRep),file),nrows=1)
        data = pandas.read_csv(os.path.join(path,condition,str(eachRep),file),usecols=columns,skiprows=rowskiplambda)
        data['repID'] = int(eachRep) ## set new column of repid
        data['conditionID'] = condition
        data=data.rename(columns = {'score_AVE':'score'})
        for eachZeroCol in zeroColumns: ## cols that don't exist in this condition
            data[eachZeroCol] = 0
        if file not in masterDataByCsv: masterDataByCsv[file] = [data]
        else: masterDataByCsv[file].append(data)
        print('.',end='')
    print(']')
    for eachCsvFilename,eachData in masterDataByCsv.items():
        pickle.dump(pandas.concat(eachData,sort=True),open(os.path.join(path,condition,'.'+eachCsvFilename),'wb'))

def cacheCsvsOfAllConditions(path=''): ##TODO FIX THIS add params to cacheCsvOfCondition call
    for eachCondition in conditions(path):
        print('caching '+eachCondition+'...',end='')
        cacheCsvOfCondition(eachCondition)
        print('done')

class Data(object):
    def __init__(self, data):
        self._data = data
    @property
    def dataframe(self):
        return self._data
    def subsetByCondition(self,substr):
        return Data(self._data[self._data['conditionID'].str.contains(substr)])
    def showEvolutionOf(self,column):
        self._data.pivot_table(values=column, index='update', columns='conditionID', aggfunc=[numpy.mean]).plot()
        plt.title('evolution of {}'.title().format(column),size=18)
        plt.xlabel('Generations')
        plt.ylabel(column)
        
class Loader(object):
    def __init__(self, path=''):
        self._path = ''
        self._conditions = None
        self._file = None
        self._data = None
        self._columns = None
        self._skiprows = None
        self._useFileAlreadyCalled = False
    def useFile(self, file): ## singular
        if self._useFileAlreadyCalled:
            raise ValueError('attempting to call useFile twice. Use new Loader() instance for a new file.')
        if not isinstance(file, CSV) and not isinstance(file,str):
            raise TypeError('file must be an instance of CSV Enum or a string (full filename of csv file)')
        if isinstance(file, CSV): file = _convertCSVEnumToStringFilename(file)
        if self._file is not None: raise TypeError('Do not define multiple csv files to load from in one invocation')
        self._file = file
        self._useFileAlreadyCalled = True
        return self
    def getCondition(self,pattern): ## can chain multiples of this
        if self._conditions is None: self._conditions = []
        self._conditions.append(pattern)
        return self
    def getColumn(self,pattern):
        if self._columns is None: self._columns = set() ##set(['update']) ##default columns here
        self._columns.add(pattern)
        return self
    def sampleEvery(self,N):
        self._skiprows = N ## mask for which rows to skip (True = skip)
        return self
    def load(self,recache=False):
        if self._conditions is None: self._conditions = list(conditions(self._path))
        if self._file is None: raise TypeError('Please specify a csv file')
        ## Step 1) Expand patterns of conditions and column names into fully qualified versions
        expandedConditions = []
        expandedColumns = []
        expandedColumnsSet = set()
        ## expand conditions from patterns
        for eachConditionPattern in self._conditions:
            for eachCondition in conditions(self._path):
                if fnmatch.fnmatch(eachCondition,eachConditionPattern): expandedConditions.append(eachCondition)
        if len(expandedConditions) == 0: raise ValueError('Error: no conditions matched')
        ## expand column names from patterns
        if not self._columns is None: ## if user didn't ask for all columns by specifying none
            for eachCondition in expandedConditions:
                for eachColumnPattern in self._columns:
                    for eachColumn in columns(self._path,eachCondition,self._file):
                        eachColumn = eachColumn.strip('_AVE')
                        if fnmatch.fnmatch(eachColumn,eachColumnPattern): expandedColumnsSet.add(eachColumn)
            if len(expandedColumnsSet) == 0: raise ValueError('Error: no columns matched. Check your getColumns() use.')
            expandedColumns = list(expandedColumnsSet)
        ## Step 2) Check conditions and caches to see if cache files can satisfy the query demand
        cacheCanSatisfy = True
        if not recache:
            for eachCondition in expandedConditions:
                if not cacheCanSatisfy: break
                cacheFilePath = os.path.join(self._path,eachCondition,'.'+self._file)
                if not os.path.exists(cacheFilePath): cacheCanSatisfy = False
                else: ## cache can at least satisfy this one condition
                    columnNamesInCache = list(pickle.load(open(cacheFilePath,'rb')).columns)
                    if len(expandedColumns) != 0: ## if user didn't ask for all columns by specifying none
                        for eachColumn in expandedColumns:
                            if eachColumn not in columnNamesInCache: cacheCanSatisfy = False
                            if not cacheCanSatisfy: break
                    if not cacheCanSatisfy:
                        raise ValueError("Error: There's a cached version, but it doesn't have what you asked for. You should force a recache 'load(recache=True)'")
        ## Step 3) for each condition, find columns, load desired columns only, and use data in name_AVE if name not exists.
        ## but use cache if it can satisfy user request for data
        if not cacheCanSatisfy or recache:
            print('loading from original files')
            for eachCondition in expandedConditions:
                fileCols = list(columns(self._path,eachCondition,self._file))
                filterCols = set(list(expandedColumns)[:]) ## copy ([:]), and cast copy to set
                zeroCols = []
                if filterCols is not None: ## This section fixes AVE types: ex: 'score' if only 'score_AVE' is available
                    tempFilterCols = list(filterCols)
                    for eachColumn in tempFilterCols:
                        if eachColumn not in fileCols:
                            if eachColumn+'_AVE' not in fileCols:
                                filterCols.remove(eachColumn)
                                zeroCols.append(eachColumn)
                            else:
                                filterCols.remove(eachColumn)
                                filterCols.add(eachColumn+'_AVE')
                cacheFilePath = os.path.join(self._path,eachCondition,'.'+self._file)
                if not os.path.exists(cacheFilePath) or recache: ## cache if not already cached
                    cacheCsvOfCondition(eachCondition, self._file, path=self._path, columns=filterCols, skiprows=self._skiprows, zeroColumns=zeroCols)
        else:
            print('loading from cache')
        ## Step 4) load from the cache that must now exist one way or another
        allDataFrames = [pickle.load(open(os.path.join(self._path,eachCondition,'.'+self._file),'rb')) for eachCondition in expandedConditions]
        self._data = pandas.concat(allDataFrames,sort=True)
        if cacheCanSatisfy and self._skiprows != None:
            self._data = self._data[self._data['update'] % self._skiprows == 0]
        return self
    @property
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

## Uses CITuple
if __name__ == '__main__':
    data = Loader().useFile(CSV.LOD).getColumn('update').getColumn('score').load().data()
    d = data.subsetByCondition('PathAssociation')._data#.showEvolutionOf('score_AVE')
    pt = d.pivot_table(values='score_AVE', index='update', columns='conditionID', aggfunc=[numpy.mean,CIError])
    for eachCondition in pt['mean'].columns.values:
        fill_between(pt['mean'][eachCondition].index,[e[0] for e in pt['CITuple'][eachCondition]],[e[1] for e in pt['CIError'][eachCondition]],alpha=0.3)
        plot(pt['mean'][eachCondition])

if __name__ == '__main__':
    data.showEvolutionOf('score_AVE')
    legend(['Memory','Berry','EdlundMaze','PasthAssociation'])
