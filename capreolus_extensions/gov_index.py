from capreolus import ConfigOption, Dependency, constants, parse_config_string
from capreolus.index import Index, AnseriniIndex


@Index.register
class GovIndex(AnseriniIndex):
    module_name = "gov2index"
    path = "/GW/NeuralIR/nobackup/index.gov2.pos+docvectors+nostem"  # store the anserini index
    dependencies = [
        Dependency(key="collection", module="collection", name="gov2collection")
    ]

    def get_index_path(self):
        return self.path

    def exists(self):
        return True

    def get_doc(self, doc_id):
        return self.collection.get_doc(doc_id)
