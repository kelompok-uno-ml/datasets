import asyncio
import pandas as pd
from scrapling import StealthyFetcher

async def scrape_tiktok_metadata(hashtag):
    fetcher = StealthyFetcher()
    url = f"https://www.tiktok.com/tag/{hashtag}"
    print(f"Mengeksekusi Web Scraping untuk Dataset: #{hashtag} ...\n")
    
    # PERBAIKAN: Menunggu secara spesifik hingga elemen angka views muncul di layar feed
    page = await fetcher.async_fetch(url, wait_selector="strong[data-e2e='video-views']")
    videos = page.css("div[data-e2e='challenge-item']")
    
    print(f"Berhasil menemukan {len(videos)} video untuk diekstrak metadatanya.")
    print("-" * 60)
    
    scraped_data = []
    
    for vid in videos[:15]: 
        link_element = vid.css("a")
        url_video = link_element[0].attrib.get("href") if link_element else "None"
        
        views_element = vid.css("strong[data-e2e='video-views']")
        views = views_element[0].text if views_element else "0"
        
        caption_element = vid.css("div[data-e2e='challenge-item-desc']")
        caption_text = caption_element[0].text if caption_element else ""
        
        music_element = vid.css("a[data-e2e='challenge-item-music']")
        music_title = music_element[0].text if music_element else "Original Sound"
        
        print(f"URL: {url_video}")
        print(f"Views: {views} | Music: {music_title}\n")
        
        scraped_data.append({
            "Hashtag_Pencarian": hashtag,
            "URL_Video": url_video,
            "Caption_Mentah": caption_text,
            "Sound_Musik": music_title,
            "Views_Skor_Interaksi": views,
        })
        
    if scraped_data:
        df = pd.DataFrame(scraped_data)
        nama_file = f"dataset_mentah_ML_{hashtag}.csv"
        df.to_csv(nama_file, index=False, encoding='utf-8')
        print(f"✅ Metadata sukses diekstrak dan disimpan ke: {nama_file}")

asyncio.run(scrape_tiktok_metadata("machinelearning"))