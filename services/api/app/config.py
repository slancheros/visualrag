from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ragvisual-api"
    DEBUG: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
