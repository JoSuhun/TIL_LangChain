from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    load_dotenv()
    print("hello langchian")

    information = """
        보호용 플레이트가 달린 자전거 슈트와 사이클 헬멧[7]을 쓴 남자로, '저스티스 호'라는 이름의 유틸리티 자전거를 타고 다닌다. B급에서는 통하지 않을 만큼 약하지만 정신력과 마음가짐은 영웅의 귀감이다. 자전거부터 기술까지 전부 저스티스란 단어가 들어간 게 특징인데, 자전거를 던지는 '저스티스 크래시'나 그냥 맨몸으로 들이받는 '저스티스 태클' 등의 기술이 대표적인 예시. 이는 무면허 라이더의 정체성을 나타내주는 상징적인 의미로도 해석할 수 있다.
    
        C급 1위이며 B급으로 승급할 수 있는 조건을 가지고 있지만[8] B급에 머무를 실력이 아니라며 무면허 라이더 본인이 승급을 거부하고 있다.[9] 애니메이션 5화에서 협회 직원들이 무면허 라이더가 B급으로 올라가지 않는 것에 대해 의문을 표하는 내용이 추가되었다.
    
        B급에서도 통하지 않을 정도로 약하지만 그래도 단련되지 않은 일반인과 비교하면 훨씬 강하며 귀급 상위로 추정되는 심해왕이 그냥 가지고 놀았을 때도 중상으로 끝났고[10][11] 원작 98화에서 꽃미남 가면 아마이마스크의 소속사 후배 여섯 명이 1년 동안 훈련을 받았음에도 합격자는 단 두 명, 그것마저 고작 C급이었다. C급 히어로들은 작중에서 '싸움 잘하는 일반인' 취급을 받지만 (특별한 육체 단련을 하지 않은) 평범한 일반인 수준은 절대 아니며, 무면허 라이더는 그 중에서도 확실히 강한 편이다. 실제로 드라마 CD의 내용을 보면 무면허 라이더는 권총이 먹히지 않는 괴인을 쓰러뜨리기도 했다. 단지 이미 인간이라 볼 수 없는 사이타마나 귀급 괴인을 1대 1로 발라버릴 수 있는 S급에 비하면 한없이 약자일 뿐이다.
    
        일반인보단 강하지만 초인이 아닌 평범한 인간이라는 점에서 현실의 히어로인 군인, 경찰, 소방관과 비슷하다. 코스튬 자경단이라는 점에서 현실의 리얼라이프 슈퍼히어로와 더 비슷할지도. 이들 또한 대가를 바라지 않고 민간인을 보호하는 평범한 사람들이다. 이런 현실과 비슷한 히어로의 모습을 가진 무면허 라이더는 독자들에게 많은 공감대를 줘서 그런지 인기가 많다.
    
        괴인과 범죄자와 싸우는 것 이외에도 각지를 돌아다니며 민간 봉사를 비롯한 공공봉사로서의 히어로 활동도 적극적으로 임하고 있다. 오리지널에서는 히어로 활동의 예시로 언급되었고 리메이크 번외편 부록에서는 나뭇가지에 걸린 풍선을 잡아 어린아이에게 주는 장면이 나왔다. 해당 장면은 애니메이션 4화에서도 나왔다.
    
        조연이지만 주역들 못지 않은, 혹은 그 이상의 존재감을 내는 캐릭터. 다른 히어로 캐릭터들은 대부분 괴인들과의 힘 싸움, 혹은 사적인 목적과 감정으로 움직이곤 하지만 무면허 라이더는 순수하게 시민들을 지켜낸다는 이타적 목적으로 자신을 희생한다. 본인의 힘은 다른 히어로들에 비해 한참 뒤떨어지는 게 사실이지만 시민들을 지킨다는 의지는 다른 히어로들에 뒤지지 않는다. 전형적이면서도 극적인 정의의 히어로 캐릭터이다. 단행본의 표지를 장식하고, 작중 시민들이 진심을 다해 그를 응원하고, 독자와 시청자들의 평가가 좋은 것도 무면허 라이더가 순수한 영웅 정신을 지닌 캐릭터이기 때문이다.
    
        작중에서 사이타마가 인정하고 경의를 표하는 몇 안되는 히어로이며 사이타마가 추구하는 히어로상에 가장 근접한 인물.
    
        작중 제대로 된 도덕성을 지녔기로 손꼽히는 탱크톱 마스터 또한 무면허 라이더를 진정한 히어로로 인정해주고 신뢰한다. 리메이크판에서는 끈끈한 우정을 과시하는 중이다.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm = ChatOllama(model="llama3.2")
    # llm = ChatOllama(model="mistral")

    chain = summary_prompt_template | llm | StrOutputParser()
    res = chain.invoke(input={"information": information})

    print(res)