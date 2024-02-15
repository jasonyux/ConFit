from dataclasses import dataclass

@dataclass
class Document:
	content: str
	metadata: dict
	
# example = Document(
# 	content="This is a very short document about dinosaurs. I am not sure what else to say.",
# 	metadata={"doc_id": "12345", "author": "John Doe"}
# )