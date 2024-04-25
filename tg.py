import asyncio
from aiogram import Bot, Dispatcher, types
from langchain.vectorstores import LanceDB # Импорт необходимых модулей
import lancedb
import langchain.prompts
import langchain.chains
import langchain.llms.base
from YaGPT import YaGPTEmbeddings, YandexLLM

# Токен идентификации бота Telegram
BOT_TOKEN = "7046235062:AAHKLp13MseK6uBClaK55HYPBwNc_1UCmjM"
# Идентификатор папки Yandex Cloud
FOLDER_ID = "b1gdohf4ce6acraojfto"
# API ключ Yandex Cloud
API_KEY = "AQVNyYxBNqu_YRTLGNlyA57PF9f840G6e9yx8T27"

# Создаем бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Создаем LanceDB Vector Store
embedding = YaGPTEmbeddings(FOLDER_ID, API_KEY)
lance_db = lancedb.connect("store")
table = lance_db.open_table("vector_index")
vec_store = LanceDB(table, embedding)
retriever = vec_store.as_retriever(search_kwargs={"k": 5})

# Инструкции для модели генерации текста
instructions = """
Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника.
"""
# Инициализация модели генерации текста YandexLLM
llm = YandexLLM(api_key="AQVNyYxBNqu_YRTLGNlyA57PF9f840G6e9yx8T27", folder_id="b1gdohf4ce6acraojfto", instruction_text=instructions)
# Шаблон для генерации текста по документу
document_prompt = langchain.prompts.PromptTemplate(input_variables=["page_content"], template="{page_content}")
# Имя переменной для передачи контекста документа
document_variable_name = "context"
# Переопределение шаблона для запросов к модели
stuff_prompt_override = """
Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
Текст:
-----
{context}
-----
Вопрос:
{query}"""
# Создание шаблона запроса к модели
prompt = langchain.prompts.PromptTemplate(template=stuff_prompt_override, input_variables=["context", "query"])
# Создание цепочки для обработки запросов
llm_chain = langchain.chains.LLMChain(llm=llm, prompt=prompt)
chain = langchain.chains.StuffDocumentsChain(llm_chain=llm_chain, document_prompt=document_prompt, document_variable_name=document_variable_name)

# Обработчик текстовых сообщений
@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    chat_id = message.chat.id
    text = message.text
    await do_search(chat_id, text)

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def handle_start(message: types.Message):
    await message.reply("Привет! Я сотрудник Yandex Cloud, который готов помочь тебе в ответе на все вопросы, связанные с облачными технологиями и автоматизацией различных бизнес-процессов. Просто напиши мне свой вопрос!")

# Обработчик команды /help
@dp.message_handler(commands=['help'])
async def handle_help(message: types.Message):
    help_text = """
    Я бот, который использует искусственный интеллект для ответа на твои вопросы. Просто напиши свой вопрос, и я постараюсь помочь!
    """
    await message.reply(help_text)

# Добавление кнопки "Завершить чат"
@dp.message_handler(commands=['finish'])
async def handle_finish(message: types.Message):
    await message.reply("Спасибо за общение! Если у тебя возникнут еще вопросы, не стесняйся обращаться снова.")

# Отправка сообщения через Telegram
async def tg_send(chat_id, text):
    await bot.send_message(chat_id, text)


# Функция для выполнения поиска и обработки запроса
async def do_search(chat_id, txt):
    print(f"Doing search on {txt}")
    res = retriever.get_relevant_documents(txt)
    res = chain.run(input_documents=res, query=txt)
    await tg_send(chat_id, res)

# Основная функция для запуска бота
async def main():
    print("Starting bot polling...")
    await dp.start_polling()

if __name__ == '__main__':
    asyncio.run(main())
