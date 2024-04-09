class Review:
    def __init__(self, data_dict, enforce_fields=True):
        if enforce_fields and None in [data_dict.get('text'), data_dict.get('funny'), data_dict.get('cool'), data_dict.get('useful')] or data_dict.get('text') == "":
            print("skipping")
            # raise Exception("Missing necessary field(s)")
        #  Add more validation here for the stars and other votes
        
        self.stars = data_dict.get('stars')
        self.useful = data_dict.get('useful')
        self.funny = data_dict.get('funny')
        self.cool = data_dict.get('cool')
        self.text = data_dict.get('text')
        self.embedded_text = None


    def __str__(self):
        output = "{\n    Stars: " + str(self.stars) + "\n    Useful: " + str(self.useful) + "\n    Funny: " + str(self.funny) + "\n    Cool: " + str(self.cool) + "\n    Text: " + self.text + "\n}"
        return output