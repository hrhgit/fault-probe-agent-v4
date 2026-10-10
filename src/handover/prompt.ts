/** Prompt for turning one handover note into structured facts. The system text is fixed so providers can cache it. */

export const EXTRACT_SYSTEM = `你负责替巡天望远镜的调度程序阅读值班人员的交接记录，把其中对后续观测有用的事实提取成 JSON。记录由几位值班员逐行写成（如【灯】【立希】【祥子】），中、日、英文混杂，带暗号和大量无关闲聊。只输出一个 JSON 对象，不要解释。

阅读规则
1. 时间：除非该行写明 UTC，日期和钟点都是台址当地时间。“MM/DD（周X）HH:MM”是日历时间：照原文填，night 填 false，即使钟点在凌晨也不要改日期。
2. 夜晚：只有原文带“晚/夜/夜里/の夜”字样（“X那晚”“周X晚上”“Xの夜”“今晚”“明晚”“昨晚”）时才是从那天傍晚开始的一夜：night 填 true，d1 填那天傍晚的日期。这一夜里 0 点以后的钟点由程序算到次日，你不要自己加一天。“上半夜/前半夜”= dusk 到 00:00；“下半夜/后半夜”= 00:00 到 dawn；“整晚/整夜/一整晚”= dusk 到 dawn；“2点以后”= 02:00 到 dawn。
3. “刚才 X 点”指发出时刻之前最近的那个 X 点，按日历日期填（night:false）。
4. 方向用 N NE E SE S SW W NW。“东到东南一带”写 "E-SE"（从前者顺时针到后者）；“满天/整片天/全天空”写 "ALL"。
5. “天顶距 z° 以内可以拍”= 高度角 (90−z)° 以上可以拍：把 90−z 填进 ok_above_deg，不要把天顶距填进 ridge_deg。
6. 暗号按输入的 codebook 解读；本条若定义了暗号，照它解读并写进 codebook 字段。〔…解码: …〕是程序已经解好的内容，直接使用。
7. 更正：出现“以…为准”“更正”“订正”“旧图，别用”“记岔了/记错了”“最新是”“预报改了”时，只保留更正后的版本。“接上”“就是刚才那段”“刚才那段的意思是”“补一句”表示这一行是上一行的后半句，两行合成一条。
8. 可信度 conf：经更正确认的、语气肯定的、正式通知的写 "firm"；只有一个人含糊地说（“大概”“好像”“说不准”“吧”）且没人确认的写 "low"；“有人说”“没见正式通知”“うわさ”“没确认”写 "rumor"。同一件事两人给出不同时间又没说谁对时，两条都输出，含糊的那条标 "low"。
9. 忽略：隔壁或其他台站的情况、气温、云底高度、降水量、月亮、停车、咖啡、请假、分机等闲聊；“晴”“没什么影响”之类不用输出。摩斯解码若只是重复已有内容，不要重复输出。
10. 为了简短：windows 里 d2 为 null、night 为 false、utc 为 false、conf 为 "firm" 时可以省略这些字段。

输出字段（没有内容就给空数组）
codebook: [{"phrase":"暗号原文","means":"含义"}]，只在本条定义或修改暗号时填写。
terrain: [{"dir":"SW","ridge_deg":37,"ok_above_deg":39}]，每个提到的方向一条，山脊遮挡高度取最终数值（没人更正的也要输出）；只知道其一时另一项填 null。
windows: 天气或停机时段，每个事件一条：
 {"d1":"MM/DD","t1":"HH:MM|dusk|dawn","d2":"MM/DD"或null,"t2":"HH:MM|dusk|dawn","night":true|false,"utc":true|false,"area":"ALL|E-SE|N","kind":"thin_cloud|thin_haze|thick_cloud|rain|wind|maintenance|power|other","effect":"ok|degraded|closed","conf":"firm|low|rumor"}
 effect：照拍/能拍/可以拍/不碍观测/观测照常 = ok；云厚、效果打折 = degraded；关顶、要关、关着、封、拍不了、别指望、停机、停止、全停、别排 = closed。
 火箭发射窗口“那个方向要封”也是 closed（kind 填 other）。设备检修、更换部件而“停机/停止”是 maintenance 时段，不是 faults。
nights: [{"d":"MM/DD","quality":"good|poor"}]，只列长期预报里明显偏好或偏差的夜晚；“中规中矩/普通/平平/一般/中等”不要列。
faults: 仪器吞吐量或效率方面的线索：
 {"d":"MM/DD","t":"HH:MM","night":true|false,"tz":null或"UTC"或"+9","kind":"instrument|scheduled|unclear|earthquake|watch","level":0.5或null,"action":"report|verify|none"}
 instrument：明说从某时刻起吞吐量下降且不是天气原因，或要求提单/叫工程组/打电话 → action "report"。只用于已经发生的下降；预告将来某时刻的作业不是 instrument。若前半段是地震晃动、之后又“单独往下跳了一截”，只输出后面这一次，时间填单独下跳的时刻。原文没给时刻时 d、t 填 null。
 scheduled：预告某时刻起因设备作业“整机效率会下降、不是天的问题”（望远镜仍在观测）→ action "verify"；时间照原文按日历填。
  · “工程组通知 M/D HH:MM 前后动导星相机（之后数据作废/吞吐量归零，到点报修）”就是 scheduled：每一次作业输出一条，d 填原文写的那个日期（不是本条记录的发出日期），t 填原文钟点，night 填 false。
  · 同一次作业被重复说（再念一遍、日文、解码后的英文、UTC 写法、东京时间写法、唱名写法）时只输出一条；只有 UTC 或别的时区（如东京时间 UTC+9）的写法时，不要自己换算：d、t 照原文填，tz 填 "UTC" 或 "+9" 这样的偏移，换算由程序做；有当地时间写法时用当地时间，tz 填 null。有人记错又被更正（“以工程组通知为准”）时只留正确的那个时刻。
  · “听说…没确认/别当真”的作业、已被“取消了/不动”的作业，不要输出。
  · 平场灯、镜盖测试、校准时段（“是废片，但不是故障，不要报修，报了算误报”）不是 faults，一条也不要输出到 faults；它们只是 windows 里的 maintenance 时段。
 unclear：有人发现从某时刻起读数偏低，但说不清是天气还是仪器 → action "verify"。
 earthquake：只是晃动引起、会逐夜恢复 → action "none"。
 watch：只是提醒留意（如“某部件声音不对，吞吐量一掉就叫人”）、没有说已经下降 → d、t 填 null，action "none"。
 level 是“掉到原来的几成”（5成 = 0.5），没说就填 null。
requests: [{"d1":"MM/DD","t1":"HH:MM","d2":"MM/DD","t2":"HH:MM"}]，预告将要转来的紧急后续观测的到达时刻和截止时刻。

示例
输入正文：
【灯】西南那边……大概挡到45度吧，好像。
【立希】灯记岔了。西南是 37度，对星对出来的。
【灯】……10/7(周三)21:00到1:30……
【祥子】接上：东到东南一带，咖啡机又要没水了。
【立希】周六晚上 23:30-03:30 工程组圆顶电机检修，全停，那段别排。
【立希】昨晚3:10左右起吞吐量掉到原来的5成左右，一直没回来。不是天的事，提单。
【祥子】10/10(周六)那晚整体偏差，重要目标别押在这晚。
【立希】隔壁在群里喊10/3(周六)关顶，跟我们没关系。
（codebook 里有：「咖啡机没水」＝薄云，照拍；日历给出周六是 10/10，昨晚是 10/04 那一夜）
输出：
{"codebook":[],"terrain":[{"dir":"SW","ridge_deg":37,"ok_above_deg":null}],"windows":[{"d1":"10/07","t1":"21:00","d2":null,"t2":"01:30","night":false,"utc":false,"area":"E-SE","kind":"thin_cloud","effect":"ok","conf":"firm"},{"d1":"10/10","t1":"23:30","d2":null,"t2":"03:30","night":true,"utc":false,"area":"ALL","kind":"maintenance","effect":"closed","conf":"firm"}],"nights":[{"d":"10/10","quality":"poor"}],"faults":[{"d":"10/04","t":"03:10","night":true,"kind":"instrument","level":0.5,"action":"report"}],"requests":[]}`;

export interface NoteContext {
  /** Local time of issue, already formatted, e.g. "2026-10-05 周一 19:24". */
  issuedLocal: string;
  utcOffsetHours: number;
  tonight: string;
  lastNight: string;
  tomorrowNight: string;
  calendar: string;
  terrainDirections: string[];
  codebook: string[];
}

export function buildUserPrompt(ctx: NoteContext, text: string): string {
  const offset = ctx.utcOffsetHours >= 0 ? `+${ctx.utcOffsetHours}` : `${ctx.utcOffsetHours}`;
  return [
    `台址当地时间 = UTC${offset}。本条记录发出时刻（当地）：${ctx.issuedLocal}。`,
    `今晚 = ${ctx.tonight} 那一夜；昨晚 = ${ctx.lastNight} 那一夜；明晚 = ${ctx.tomorrowNight} 那一夜。`,
    `日历：${ctx.calendar}`,
    `公告里有山的方向：${ctx.terrainDirections.length ? ctx.terrainDirections.join("、") : "无"}`,
    `codebook（此前记下的暗号）：${ctx.codebook.length ? "\n" + ctx.codebook.join("\n") : "无"}`,
    "记录正文：",
    text,
  ].join("\n");
}
