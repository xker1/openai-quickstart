from book import ContentType

class Model:
    def make_text_prompt(self, text: str, original_language: str, target_language: str) -> str:
        return f"Please translate the following text from {original_language} to {target_language}, keeping the original simple syntax: {text}"

    def make_table_prompt(self, table: str, original_language: str, target_language: str) -> str:
        return f"Please translate the following table from {original_language} to {target_language}, keeping the spacing (spaces, separators), and return it in table format:\n{table}"

    def translate_prompt(self, content, original_language: str, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, original_language, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), original_language, target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")
