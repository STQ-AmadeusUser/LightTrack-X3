from lib.utils.deploy_helper import print_properties


class ModelDeploy(object):
    def __init__(self, models):
        super(ModelDeploy, self).__init__()
        self.template_size = 128
        self.search_size = 256
        self.stride = 16
        self.score_size = 16
        self.init_arch(models)

    def init_arch(self, model):
        self.inference = model['inference']

    def template(self, z):
        self.z = z

    def track(self, x):
        cls, reg = self.inference[0].forward([self.z, x])
        # print('cls feature map:')
        # print_properties(cls.properties)
        # print('reg feature map:')
        # print_properties(reg.properties)
        return cls.buffer, reg.buffer
