import json
import random
from time import time
from lxml.html import fromstring
from lxml.html.clean import Cleaner  # remove javascript
from multiprocessing import Manager, Pool

from capreolus.collection import Collection
from capreolus.utils.loginit import get_logger

from capreolus_extensions.gov2_utils import *
# DOC, TERMINATING_DOC, DOCNO, TERMINATING_DOCNO, DOCHDR, TERMINATING_DOCHDR
# clean_documents, parse_record, spawn_child_process_to_read_docs

logger = get_logger(__name__)


@Collection.register
class GOV2Collection(Collection):
    path = "/GW/NeuralIR/nobackup/GOV2/GOV2_data"  # under this file should be a list of

    collection_type = "TrecwebCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    module_name = "gov2collection"

    BUFFER_SIZE = 500

    @property
    def id2pos(self):
        if not hasattr(self, "_id2pos"):
            self.prepare_id2pos()
        return self._id2pos

    def prepare_id2pos(self):
        self.file_buffer = {}

        cache_dir = self.get_cache_path()
        cache_dir.mkdir(exist_ok=True, parents=True)
        id2pos_fn = cache_dir / "id2pos.json"
        if id2pos_fn.exists():
            logger.info(f"collection docid2pos found under {id2pos_fn}")
            self._id2pos = json.load(open(id2pos_fn))
            return

        rootdir = self.path
        manager = Manager()
        shared_id2pos = manager.dict()
        all_dirs = [f"{rootdir}/{subdir}" for subdir in os.listdir(rootdir) if os.path.isdir(rootdir + "/" + subdir)]
        args_list = [{"path": folder, "shared_id2pos": shared_id2pos} for folder in sorted(all_dirs)]
        logger.info(f"{len(all_dirs)} dirs found")

        multiprocess_start = time()
        print("Start multiprocess")

        with Pool(processes=12) as p:
            p.map(spawn_child_process_to_read_docs, args_list)

        logger.info("Getting all documents from disk took: {0}".format(time() - multiprocess_start))
        logger.info(f"Saving collection docid2pos found to {id2pos_fn}...")
        with open(id2pos_fn, "w") as fp:
            json.dump(shared_id2pos.copy(), fp)

    def get_opened_file(self, fn):
        if not hasattr(self, "file_buffer"):
            self.file_buffer = {}

        if fn in self.file_buffer:
            return self.file_buffer[fn]

        if len(self.file_buffer) == self.BUFFER_SIZE:  # close the file with oldest use time
            del self.file_buffer[random.choice(list(self.file_buffer.keys()))]

        self.file_buffer[fn] = gzip.open(fn)
        return self.file_buffer[fn]

    def build(self):
        self.cleaner = Cleaner(
            comments=True,  # True = remove comments
            meta=True,  # True = remove meta tags
            scripts=True,  # True = remove script tags
            embedded=True,  # True = remove embeded tags
        )

    @staticmethod
    def terms_in_line(line):
        for term in [DOC, DOCHDR, DOCNO, TERMINATING_DOC, TERMINATING_DOCHDR, TERMINATING_DOCNO]:
            if term.lower() in line.lower():
                return True
        return False

    def get_doc(self, docid):
        """ docid: in format of GX000-10-0000000 """
        dir_name, subdir, subid = docid.split("-")
        f = self.get_opened_file(f"{self.path}/{dir_name}/{subdir}.gz")
        start, offset = self.id2pos[docid]
        f.seek(start, 0)
        rawdoc = f.read(offset).decode("utf-8", errors="ignore")
        doc_lines = [line.strip() for line in clean_documents(rawdoc)]
        id, doc = parse_record(doc_lines, return_doc=True)
        assert id == docid
        try:
            doc = self.cleaner.clean_html(doc.encode())
            doc = fromstring(doc).text_content()
        except Exception as e:
            print(id, e)

        doc = " ".join(doc.split())
        return doc
