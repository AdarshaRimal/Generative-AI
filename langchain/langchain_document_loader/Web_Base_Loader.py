# text content from webpage
# internally use request and beautiful soup
# work well with static page
# for javascript heavy use seleneiumurlloader

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader('https://www.daraz.com.np/products/gaming-mouse-mouse-with-rgb-led-light-2400dpi-10m-clicks-wired-gaming-mouse-i131291721-s1038146149.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Agaming%252Bmouse%253Bnid%253A131291721%253Bsrc%253ALazadaMainSrp%253Brn%253A6c4a1349de362e3cf2d8ec66a13ed19d%253Bregion%253Anp%253Bsku%253A131291721_NP%253Bprice%253A165%253Bclient%253Adesktop%253Bsupplier_id%253A900201712735%253Bbiz_source%253Ahttps%253A%252F%252Fwww.daraz.com.np%252F%253Bslot%253A0%253Butlog_bucket_id%253A470687%253Basc_category_id%253A7849%253Bitem_id%253A131291721%253Bsku_id%253A1038146149%253Bshop_id%253A122507%253BtemplateInfo%253A-1_A3_C%25231103_L%2523&freeshipping=0&fs_ab=1&fuse_fs=&lang=en&location=Bagmati%20Province&price=165&priceCompare=skuId%3A1038146149%3Bsource%3Alazada-search-voucher%3Bsn%3A6c4a1349de362e3cf2d8ec66a13ed19d%3BoriginPrice%3A16500%3BdisplayPrice%3A16500%3BsinglePromotionId%3A-1%3BsingleToolCode%3AmockedSalePrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1752859921892&ratingscore=4.8311688311688314&request_id=6c4a1349de362e3cf2d8ec66a13ed19d&review=231&sale=1197&search=1&source=search&spm=a2a0e.searchlist.list.0&stock=1')

doc = loader.load()
print(doc)