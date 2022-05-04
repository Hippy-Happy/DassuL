
import discord, asyncio
from discord.ext import commands
import nest_asyncio
import os
import json
import requests


API_URL = 'http://127.0.0.1:5000/pred'

game = discord.Game('채팅 검열')
bot = commands.Bot(command_prefix='!', status=discord.Status.online, activity=game)

@bot.command(aliases=['안녕','hi','안녕하세요'])
async def hello(ctx):
    await ctx.send(f'{ctx.author.mention}님 어서오세요')

@bot.command()
async def hihi(ctx):
    await ctx.send('huihui')

@bot.event
async def on_message(message):
    # Bot이 입력한 메시지인 경우
    if message.author.bot:
        return None

    else:
        # Flask 서버로 message 전송
        user_info = message.author.name
        text = message.content
        data = {"UserInfo": user_info, "text": text}
        response = requests.post(API_URL, json=data)
        result = json.loads(response.text)
        print(result)

        # Flask 서버에서 받은 문장을 출력
        if result != ' 이 문장은 깨끗합니다! ':
            out = result
            await message.delete()
            await(message.channel.send(f'{message.author.mention} {out}'))

bot.run('OTY0MDMxMTE1NjEyNTM2OTAy.G6ijnk.yySn3_gbIV81f4uqME_d-YgvBwKnUi_tiICQVM')





