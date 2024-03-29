import os
import sys
import subprocess
import shutil
import tarfile
import base64
import traceback
import io
from textwrap import dedent
from contextlib import redirect_stderr
from typing import List, Tuple, TypeVar, Generic, Union, Optional

from collections import Counter


USE_PYPY = False


def build():
    shutil.rmtree("src/include", ignore_errors=True)

    os.makedirs("build", exist_ok=True)
    os.makedirs("src", exist_ok=True)
    os.chdir("src")
    os.makedirs("include", exist_ok=True)
    os.chdir("include")

    if USE_PYPY:
        with open("xz.tar.xz", "bw") as f:
            f.write(base64.b85decode(
                "{Wp48S^xk9=GL@E0stWa761SMbT8$j<p2Mpzg++q3)}>NP%<X&9eo?~^2;;q?D#kE>ehQX!Mqz*9>WvL4T%%^AA^kmP}-HgUgxoheHxL?_!_uilpmS6YAKy8eN`z0DP^<8->J>&Z3%k`B6x0GylttquVZJy=R>17rpU7`f|mFMg~8f<3Sv0ZB{LV=V8q=1rvyoPrdh577y8rHX!9eF%Jt<u#4-#RUI598#Iz%+0nGjyk&wEH9+f&28@P+>D=2(lk7g%LzGD4q`?b{SahqlH7#MIB^faPNB@ut}m!3)4y)qNf<4Xw|NZLxf8P1X;s26`V=WdZmMKLdZvUt9QwtF7*4&IzL<uP>teDA7h@|#}9>=#qNW1y8SYHr>sYs7V)P?Y5>Er3Q$yzg7jw9kdOW1uYirx!Euzl&*P7d9`a&Qd!~=`0aPiMY2<pu?J)GvslQLK3!0L(WO_7mo(%>H=o8Px$Fj`8n_+O4Ek+TLgpvfaP5<>50qpCdnTtUYjDX{EOHdnnndZ%r$Qk^9jNSkqK`_jBf;!%|jipnsF`=9@C-c`t>`QGoGu$Q4W;nEz0l$f~UgfY4_(y#6kkDLJ2&w*NsW@IFitfR_iOuUVj|aCAGhtQer%+WOO7|6je7pWfNu{$I%Xn@zS0uS46>3jkSDK9BD09ad!v!Z(Xn0)%_~wcAmcl+V|}?78Hc5Rd}>$aWa++LlaCKYvK_j9144{T~5+NW;1^u3U}>QHUV5L!6)jjoBJco<vhl-bN*y-jk#>wfQ&z{USlmnfxeVlJ$NFY+*N}2F<6;06K4#dhxr9Pl+4(d0~{j_7Q*z7JN!P~?s(YZhKidQweC<)u1Rr`dnB|9tv=;1KylYZs{}z8c6im_>MSCe>GGT8fX9ZVxsf=5kDsxG;SHNO{Z1cfc`KqQ^#e147I8o*sjnjp#S?+wUdM)JJGJ<y4f<uTyd`c*XCw`1x<Tn?%qpZWG9PqsO#9XI$kx&*(`ogtIA>3uX3cw<EsRr48CAa>SO^Z8$q1jjhu|+<0DpU9wlr`46ve;|2(7#pxU|)LZcsu6(8Cfrre)%0Lz@>Ms*Uh;45D{u4`b=Ae#DS?eHL6Gu!l$XCCvtARe`Z2AX}Mfa0a9uYQo8$CQK&*&B!dc0a(JP#$dQ(A0xqZZ_&jZKB5?5M{%1ypgJa=su&?G^g>Qj1rO*D6<{;#<>`I@hbR7SlSE-9Mla@r^Xlnbb$~c_OvM9x2p&}Fv^1|^N3~;XW26URj(7R!?yK8xp0YFzGn%umWdaiq`Gai5s9ol>K_X&+INVj14%qdz&_pyv_DK~ihvp|lZmAlor{2fkWTpdJ!aR22))acb7WZt}CE4%A9Av7|3)SgV5^-FoYA%<TO8u8HWJ5%o3ZLVXr`xD_<v*P`Lj0{nH`{@^5?j@;srZ7xCnX5YPKvYYOq}aH8%Lzd9hMf5a?$zFV@47_|88Go@o5<T_wNenC+hlp5h1tnoKT<=_>#2lrj-fS*P1BxcdZ-DpXC3BX>x@uCdNwHKjZo_kE5qDn|)IhtFB2=GJZ$Z2_q+DNv;Pr>xu-~<E5(<d$!Ce%eYK|z}2+sfz$p+(UWUl+i9*TS3Kt$ay<0We|Cz2lnLHZ*viLyzbS<?nSaw=zsmvB{V%Je2DdVk3XiswL3g!5mLW^&m*9o;b*JCyaoP3}8}C1q2XKgs>K`QG|HzxjJ#X9tTG7?p0v;5hWExbaYom&{4u~O~tD9pbYp)qf#?=<k5LdKp_$;0bS28=-SWv6QF~on6DPJFTcHXz00IPvk4G5@oZeE+%vFefZ;!QD2%9}oR{GJ%r;;}hBk-US?$~Jvni7iy&7}|vEuwN!iEZf^;An!tvuh+qpGB*-WqKP*&AEgwz!G|4n*#G3kB^xU|0T&g+%eMOQYS}_s(d$S0d*z|rgU9=D;TY_q%vv+nz<!sT$cKj?8Wu;XrKB4AY1*Dzg$eXdxfl@!NTqnAF8>J&KU_Q}XM)Sv%%?h(s-hAZk(8P$oJe2W^mebT3^bv7N3TA3AsXSzN0a7rTNSE!eQ4h-ojoW8I>1j*4Y8}vrqfev#3vuaXS9HkT{H1@i2cW$v$Sf=7g`s?VBg*}cARZ|sxF6;H0y0U;V3K}>)TCeLlzwXLqwk)0K!K1Z|@%eBEfIi`~gKfu)cIZY5orP+y}hAT{U?1>WH&w68#SX{Y?$svX3m0ElU6Ejlpc+8vw@ITVl_XN(Y_L#8q2&!W}wul+Jrg_d4FGM4(c%gr?Q>4;zX&Cy5sl0OKO`N^BR75!H%eeOI{+fj5|jI09CHqX{N_2CwNKCxx@X_S)UWXdm0X@IE<b%n?IhQu{xXVr2f~XwG>vqbLJPnC{ABMd4wSA&$cu7aq$PVk6ELSFmt$hx(F#=WTkn$HD!X%U$hF%jQVwWL%s6U)85Vz7<ONR-iG+(G5i$Nqko$7NgolSH2vUj~iZ)?1bLxbfQ*auDP!eh}YNl?Z0QUVwawi>{?kgP_{8T>dQPevtRG|SSy|a$l`*(_!?2r;B#?9#8x`ZJglmFKQFW%`Lj<+A0(X})?&{@+#wC6k(bEfrrUB$0zb=1ze)8zqdLScB-S3-94q)Tc-3UiL_zWRZAWGf8gEqjP0Bz|?CU{^^TKkvQpM<FcZB+i)et7CVQKQ_8mfyqBb`Tgk2u0s@UHDbj<i+r&A+&RhcstOFEqzXj@1CdE0F=`Ldiw<7e4-h4jgoFI4IRo+X(nDR?3M~-gOk;&oB9k05uZ3*UjC9IH6}d|3el>R`xqfDtLG_MDYSf`gLQuuvu=-%P+Q}@`j5f(9Kz`DIpCXacGE!jDFI3MZ@LZi4wqTD@DRzD}2}8u#KGA4U$$ufAN0TNj0$n@o7`I*QVdGdfQs?swQ>E7RtRBvd6&sI$Kjt-4v5fwbo;eKI$zulT4PvHVu=&>Ws%bNT}=Qa7{rhG7Hw@Z;@2$loMhx9Ek%xHm_JHhJz4jOvj27Ld;r7s})BaI^eZaRwX}DAzboE4>DWk_z~sqa}rUZw@mI=v*u=msdG^yo>REjXSyn#_9BFvno$Di?BpSib4n-dXw(F1+V)6NyYq$tPE{CqI-!RRT<1e7@}b1al{QM=>RU9rrXp$cFmYbce-?Wm#8%k5(<71bQ%`RJHxE+YXkh>4WCO!iy{+P^a(w+58|#Xs0L7mBH^VA`|47LP`R-sU!>u4_cq^!1q%TKHN83E{+|Q18uUa)FEy?845GHewf!jgOd+Lb|s8$3U>Jg_mJ|Sg6YqZie>tt5umS$@8HtKc{Fsp+U1mmq!Uu9Ge6XSdUWq<jDeX)mfJnCF7w3nUy!Y+gL^IsK^!l`q64g#~mEYb9L$9xM#JWigF&m7CCCs*y>KmsFpga|va3UjhHFo=Egj|~7)g22Q${qdo=TI`cb)GZ|!YA+YxpU`&cO<VopkDF}Iq)5ZG!6fWa7uss~cJwzW(yWkFSUrZBVaBC6HpRc^_(GiAYCFQ_wbd8SGYhwt4!S+TX9!{xQ*!#(af^k7xl*;G@i})Bc)=I3<|%fdK}-A+sqF7eflqtdb0LG?*S<%$)hRTWaR`=Xfb$0VWP<?yo>2VgTW=+{*3p5^Wr+!LT!Z3M?C2Aik<n!szHqX1YdP7Wm+P&S!AzoxL)F4d2UC|}Ta<fhI4#gk0dBfKKg^OEJ)+PrzT`hJ=<|2<<o@lkJc9L32D%vliW%GU>Sk$Xr2`0T9+@PDgFA17F6P&!J8*2&&<LH6Rzo+R_p1&#{sU~}8By?K{ezq9E)u6QJl_?uR1DEIiRf+w%zcYp{r_a;omO^<A0PsETJ1EN%vlhU=&=&vOdqhmd%u<v46WGk!Tw)@Sk=K_b}zuaQHj`LyL@2=x19EPFsnt~AJ`7n$#RV=v(md{li+w0sk(q1f630#MsXI?K;)tTF#5*fc6iVhUva&xz806Z=~7}B@F4LnvUXS9fwGoW42{!*jf=iYa!^2c_Ui-~QT_i+Y`2ExjzBdlft<Gq#~#t4|A(H`3SHTK?Q7M0#omNu{#3YT%nk(e<#hkuhwV@=$}|a=;`8-!Ne~Fb#slVx(|el5g!heB_~L0X4@{Bf%ehDU>wt>_iap?ZO$b4h58&rdIx(DX>sz0!RZFF0YN##V35d=m=5Nk)K{u-;<~$S&A#vT9!3U92g-NhXkK)D)JzsfA5~&I@6tPfG9+*MsJV%e{h_^|#kk?;0sEmar8OGF+Uge=(ZTDiH@i^Mv{yu$7!^Hx&pO{VY36#{EsJOY%p%3RVFMVZqY}E3ze`N!d)G(^)aK*knH;OBzP`~!H)x%v4RyHI8D>-c8b3;`<b0bNaqc)8~n2>ru#RcLN+ut{LtBi-kqjhL#m1I&;EP^e;fl)yG=(^oOel<wo2qaKTUdh}9fH9~d(edA14G}DJv{&HQ!hqNY(+d6%mY%6|lH7cw>F&<!<#<J%R*xx)xwPr|OL)$Ry}5--0IarakUf+vA0aUVG3C;^plqZ>Gs4mi#UYiD!je@Q;@OWsu)FA9M4{-M+~d9LgK6s6`74OXiTW*~e?E(dLcS|fpbx&XN_Hf9@55^lHXWe(<^cfLM~p9S5j5y6WS_jcH{fM-sT}&`qV}x9(aQjo)XQKvS*l`$V^#JG8X=VU7^LY42p3J%h<7Q;0xG!)nQCq^|Ir~p{>oGQ#k(*Vg}cP8Zp5Jvgr4t}H{igjS=hWGx6;ojWpJDK6(Lg}MRWV&RNx9P6raPaPkJy>-gTWWlyHpE8^nm}gPOn~t@=gZ6`VmwQ^78a>0htUifj?eeasnJlrY&cI#A!VWwL|Q&;H$)__J3x%j1j@=l!6*mD?2r<P^X~@btAzA^!H_(8+#~ED)b@cwkFoFgGj&iILPx?mv^J0OQnVJG)zz$Ah$1r|?&C0wetfY|-Xit7vD6VdAb<OIw^sg>^*rX)6wjQ;m}wdX<@Ug4diM*Fz4qPR3aZO{*_tD~>gAXnu_M?PZC+PS_<PZwn*@p|YgN5`^M3&-C(Xf3Q5<T^wB>-C6P0{N890RU7G92hDeSH%$eMEh@HWsj=mRY;_u7oWzJ(u97X3Fh#tf7M9ej7jReS)PLGKns%SL{C5zu;EAXslC!a#`J`9pay>634L3fqa~<;4(7|XRlGK_0(vnBLEKW0>Uu=-7qsHwnxWQ2T${`7ghiLhdMTPHp4L%51rP_z(us<a&vn^|ZtSHAsiilH1r97XK2XijDGT~p^BSw85ws5;93iW%?F<#Y+5n)CkoQCPl;i=HE4qb@>B}u{PUwI`do4C#8qHDi7v(*1=oZ(v|qxQ3$Vn4n-HVkloSlp_kc+RDRP7Z^@i&C8l+Yf2s?V1eEiVgC7b-dX|_)BNQuRexML#(mw?OPP)Ei~88B+x?;M#?Yq7|1x*+dp*e(1goXr9EYE@yy(KpYYa2sYk(CO53$|6WHYBpbh^^09aEa?8R)@Cn2a!*AB9PJ@X7`sQZAR!FfsJ*xdD)1Xg*((81Ll1I%AJTqht3dwO&Col?yl)f<t0m{=A=i^nUWVE2cV**%lI{dP57H&$KS5r=wgkVme%?|>7n0?f10R_Eb;vVzEAF0zGl<KQ099gBkpdoz>n_KPZ4`b?SS{o0}yjsb^(N;r^Ub<KZA#zjjfprdX0(|5e|3wj#=lgwc7x)T1AJ1;h%02%V827XZxHTG|S?FwW2jENBDT}~DyM<h~kC7S~*Bub>o*>a|A&J4%;Rl!muuTj;R@^xzF{IwmJBkZ&cPogd6N<syyMFDh7A!;m{U13%>DKDu9(cJ)wiJPbo;F(Sg>8KN{rN<5*Cmg%{4y6{ug?>YaD7KTEI(0uPeYA9bvzm;@L>k7ZDMoIi6E>SoNxKt!kWJptZm})zomnvlMwe{0p?NF30{aRkx<C=7@IK9tQZDngQwm+AJ@vR_>xw4MX}4-KmYqmv@U+<)M0s{~bKu=6<S}zQ)n25T{E!GOoC$0YR8VsttDY1r%uY~6W{<BX$Job8-g~oQARW5Ktahz{Hb$;mdLam)7my9%qq{P^tXVb39A`Z{g_tUSpf)B(J?a_HF|k_d6uq#D<GIkR*0qs-E{)*->v2<|ky-V*@AIBReMf<sh|E;HaVxp0cX^o%>@*f)<+jKyZDOxFe6R2UoyQ^u<{hE_>Izt6ABH;X?s@K20q4VsNkDyUHpxZNL&>9ZPoY<}=Q+t}g8ikneAFD!HK>Kcel584Q`G+^p?GFA1p)?nVe%j^{7~zNDq6ZwK|n87HMd_8sEzTk*b<#}ioIxIG^5I*bXmvewx%;~lb}uH&AD|Ym|+CVtmUqspRBwtd;ZgoHNJ4^f&~R{d5H?Up$)qV>rdu#Cb}(lYu_PAXe<opOEKM6Gl<>m<X74pVAwe!31;wgIN<1hd6o(bKI$x7y<^>$Eywx0iAd<4($y|0kVDuC7nn}+7d)oCc}m!WvQumGlNIh}RHEYE-tfsPsw3d|GV75N7}Wvn+&C4Y1>*(L(=m~DLGD$t-eFgq1F1kchh7yeJL2ZM64#;QdN=|5o%PaY-tv+d1t&Qw3a<F+rpK+H#;%W$YA(nI?zrZs2LwU;DSMkSvQ2dOLH#{#pASGJF?vCHxe+ANj9fD|PQW{|11CyYyOZRe{s3eK5kYaLNBL;~G89#$SSjLvSv)_UhbYA5nA21!>+Z!PBUzsiA&OT<uRp5ngzyU;@7|?qU6R9VHC?dbffZ`!0YXw*762<?QGM>_Y9%bI+t5)cX%y0F#1w;}AV}b|OHG)DxFI<8INt}NuFC+5Oj}~$ORS+btaAg8y;oMxED@Q9J_qekYZmd%RUk>=6Bbr>mnr+LMd2(401E0(y_Iq1b*Vk)prdNd&;klDw}{fh=)3K8Li1IZN?9YLjE?_QEp(>0cmumznB?$!rdwt;rh&O_i5G4v8v(^v_`ypiY0piRdKk53fBFUd`$#5FaCveIMI|YhH`gE_gW=r^C?}7Gr)}kjL@~`Rh|VtgUBmjSL+YR^`&o=d@b_OTViil*d(qO3s4@HYEs>SS-*knHQdG<+_3-jn3NooExd$=X0&BaFP5t5rh_l3;T>v~$8zRgCb7U!N&+6G3ZdhIAA|Bk;?Q-eqU&KHdgE;JcoZ8sUr$2sl6J+^NpjF1C9bS<4i3alK6BvBKYZ2U)e`~D$X86H-BP5K4k1#c)4QHdBSpBAE1Iz%-+~yT=f<Xjfw@!?DwQ{pBU4|3IKS#Flj`QM~v+bS_vy=7?NLslyiW5nqG9IXh$p`ECs@4JdN<5vXVi@hvJ9e|C%B_jwaNNg<nQ7y|Mj}#KEbMsDu4cywpw8T-@&glxM|C_OK}}EFu$MR9)lAklK$A)aDd#BeD%lA3e1&AVh!f8XOgaCdJVLT^0GNf^-H^P3(X97PxRdF;yLI9oefhlHEt`qkND^+Q_{eZweTk?)2p1ss&%deml>RH=auiWx{nWJ-Y`2U{7`-B0@NM|~ru|;0Q5Z)BoZyfd3<B>Vy>L9Ev2xC}6eoYM|L2`TPJK2X>(TvWl)!C9@Iq7BJ>=J$vqqpCJ1Y|tUn92}3xs2VkqV5}WP9QB2Ktl6z?u0$jSEFbT~6*hCLQx!*(8iE{~KB>Ocdcrq;=Ag`G*80xoBkE!E_%*eLuni@ZI*%Y!o~OX#(;)5%F%F@^7-r7<sENFJg*7vL99fuqEqKlDQ%9EU4O3ClW(k%an$U21y3`!A43?jv*b@;%9i*I=*U2Jt#l@0+t-_^fRf#y-np@Hzte%78A@4kw>2-GybpAa$**Ex?r}osGNO@xecwegxF3J50_)nYJ=zA-cx)x$GnBS&TB^+9vVrt3GTfaQ2yiu92|pv^$%&P{;{^JX_dB964l@mV?s!*QrfvCG?eB4(MLKI#>xf)Z0$J%clWwGb=WvNDPM5eya4!|=6dacO8C6Lyi~V2MC?m-PD~Hb3nOXlR55e>-B2DY)Vuz1+W$hF^{QEc{Ke+7pjL5kV+%S9%;4`Kt@`!W3^sK`#eUkggn{y>$9tW$=A6-bugLGiVn2nailE0=?2$1;LT~UUID7QE1JM^_q6m~$$q3kO4?J?oAV-XyDxP2&LNOb*=_^-3^MH=SCV0Ba#XV+L{OqKvw>Zx}_2WU`tM^5cK7dQ*=pjec&qnfOVuS&Q;kcIeIg?lD@%&w+=m&}$SCRFx%f!EX3hiq9cmIa;Y62yaxPMDkfWrmFE{n@m@C&wlY+JckN~NgRRDkx_SOCnMpw6bXrf3?>1qBDeNyZ1@d_x&_G;0s2u*Np_B^~8B6F$yv&AM3=QgVhUf4?<5{w=IOu7^UHN=2-4`SxS;rh@XglVpz6gpshxARR>_CFK?}`nI>|cs;qHDEy8sU_L_Up!%Y~DgkZ^27QmV$wLv=Qd2ojmNGHp20MggICu1s|E@-im1^b}YuGmfLd4C-nG_g7Wh`WAsor8p8gd&ve9p}~cR+o&7IP%C5x!4hEKfr+$|ip^!=^M#QwTn-F(&bh#l7Y_>>Lr3qMBk<^La##D}T&7&wC`C*RSS=w}PB`?lMla?RU>4tF&m7kvk}~hk0fs@}I-3r=%@<#JE8{0Gjpv!hLw7qcHb!Kr1q;sVmSHGz_<FRCzu@pnD$>Y8V6#Mw@NkgOYN?40n;U1U~P^e`|`k8gb3@Z<dy0jxGD#0CDNRBeH(q9Ez)E`7YJs9Xmf9j|Fvh0I-0C-R{To64uPJ12UXgLM;wV7cEs&$(w&O^eus`^q1R&lzr@RUvRE?O4k#43Z#)+YaTKmh`6cCIWKYv!cm+*u!h53ruBE=H@1A`X^^5{;lyF-m789$a#zi_rn_ekMf(vm%_xh%AmTY}bgKmha}^r30J(m6(pK6*y8XoF2wd7Ee0#*kDSYWyK*!J}2C|;P{1+N6cL4zp!BLC(KyC3xA6OtYqqy|-sg14YGm}^64-+i%hH|tBpDKM*24g?!W~Wx$Y!c->@#I?wl9~j0e%?K-Qq&k`@trM*bc3K0toZjY1S@M1;W&O<)qNW+_7UEG?0~k*w9&<}`yGW4$6YhE$(iddn_vW{x=ubA^+jV*5S%?8{ijS9$*Jz{@qWMDh{~g<qQl`qw()A6t)%LLAOX-Q%DipIg0Wg;&m(`8${1Q{ydEL_fem?YEGePok1qg=T5UXyrVg=@%y2LPHJGpmlbjP9>A5n{-s%(wK(9iTJ1Ose#Ot_u?>y&XR+ls;=EaGz#yVd=77s3svE2_XsUx50+m>#0ilj<D4|(**3`0pxtVJ8inDIMhOxKO^u{R+Rb*321G9p85%3?;qKDX2w<6$*g!a(;#ko)g~Zc;z;nxxWwT}^f1Lk0|gPFC|FJ*R^p%XdC!n37L!bRn!~hFKPgQcqeP;}&Ff;nY!9hGiF}knyZ;ck(ML`vijUY&4gi6%wd>MP%+5y#J1CpMt)2=fT_U{4}BFp3=<cU%rFw6rZNFkeqJ*5{axwrukP3Ni1Av$HAGY3l2B$igQ>KH=kS3GN+9wE<Sl_Mm0$IS2zzr$U5R@0Dx@H6fFwe>83*OtESo|qgk`D_L~eYRz-=NsV-aOG)T8dgu&`AZV4F1#eXC!b86EK5?b%+G+~P62?P3y)lX{1Hotk-MV|!;QZ^A~L|guJFJzqhkQ9LJ-nnGDUlAu@OWmYp$s$eSF6ocKi>+kCk;FuH>n1590XZ2spI67{3fZ%6>@Qj7myD?t813M$@!_}GK&~@V;XWveODpViC?g#iYK~u{-5Ln5Orn<O&G){gh=X;0(CdJomz|%ISn}*LnfG+5pxHp%MmW=rH5!{f5W&;zmP7m>eX)rT&b)4*RJ=3pCaGxwg58lQBMTc3sM{9pB=cJz5^YF;NCArR+<)FLXxRc&09zrahgR6;oLD4ms`#V1s13niK5B+=uw(}3C_Jb-mx=uW!eIE<Ru38oZqU!8?a8iL(`WdM&&ln|tBBoXfT}Gz^LcBwqKY@@7grgLN|H?KhF{3-eeFL4x^{@9TIPJ8;C-fxN>{fNZ8MC5Vx5X(K&y<9e@QTY+3ZxUk!))!sJGuVCfTo!#&s*?9TE;X(Q#8DPH*rbfc#~_(wiWRss*R1q&UCoC<nEazhucRrKr=baLXh`z{jw^Y7^lN%Zjz}9hBFrWmaX^xoSCoLs98R!X(N2Hkgh>bmsM3c|hu+aR7|YcJK@;D+ECKFi&EXPAkCitbyOfH)G67>19B<ShF(Q;`xi-fKOKRQG+A%XVknQ;^j8YCZip^zsI++7K8Qt*4udN&M_*SjllELch%-NO!+4Yw~Gz;T>Eid<*k_X8px9iJc_9|as;qc6&_Kag-hTOeJ0W60=ycLP_D=e9zsX6h99kusozW??Z$E1Z3`NZUS^CAGK}4ymTu!sP1Pn1UR5IvkbvVI>KsB@R2*-{sB`;Eya$|DXZ5c*5rPV1v<;86UvkZ&^m8S8(r^W3C+$o9QSZNA-GtoNr_E|>q~$4?ztCKC1yImTMR;dz(WbX9&0Z)wf5bO`O(|L#Cw7=52>sq{qgXD^eI`oeWKLo0Bp^ZPz0m93SH6C76N_vX(Ia{u#Q{g4YtJk@VDj=u`<3_Ulh@I}t!1h9PKKX3gjqAgaoAhKL+azASNx;`4|=fXK>m<yDDI=PWE4>Ds)^B@pATJbG}<V$!)HE3YN_HXx)v={q{%@?eNPsUBbag+`CS|~ZLG2C^>>c3sX2gdhc}}0U#84cMh$Ol{slDE3eh=?q|=oFu1`R(_x96CMD@M8PEv@GEp)D$&=w5xnMh(fvD=*S>Y<3?1tfkx-A-8hv0B)*vG%787jj->6#n3*gYM$>!lY2R*Uqwok%TSM@*DLrI3a6x@*5TlOF9tc8M6hK%{kQ7Oc-WqJ8MopIP_Lt^QftYHK=1sNZsomgPb8wQcw$(!$}Aok;VB|vT```wcfg%MV|x)Bi;P>f(ad>Xt`|eC2IOFK=>f=gMoJ6VL1++WaB+vpWM2zvKvv4b~RaUlsM0f^}4+ZPKs9_8x&=jx2FE=TirIFz{8;R-ry`*`hOP0dZHZzA2RXwI3iewZA)!58)(4y^sTa}pzS*OKWpC63ICp>g}M`~Eq*!nrG}K4+@=T7LYADfiwV^@2yE}2<3EGNtH8L>Ga%36VZF>1`uL|^4Z(Y@hVBvl%R^6wIBXjnhi-e{@+)#t20a{3;H~XJ>x*%FQ;sYA2>}#1dGuC*46BK08+y|*f0}(G**4AbteX}M49#!nS5404Eb(YL+)+#mbCg1-n}+#?3rh0nODjjSRm(OCFKP%B4e|0mA0)O6TMP=EDtnJT#<W(G!tn(=<uKSIkC^TUp0*{qk?8yCAo_^PU5ir--qUM%=O{2~%7g@9zI4o&2aG~WS5m2@_dU9?_26jdBNXW2yxKazLvCnnotL`k`V{bin&Syne;L4r1Sm%N=~g>{xkYYx^fuW@lj?57=A=YSeIsj&GX_hCkd9ggfzDS3Mf|ETv+~K!r2CWsuGQW(OkIxnb)+`?D%?$RxUXT}Do!n>{TFgykCwETZbtW~&*BlX?Qh)CSlbgkIJLQGK94N)AHEk@(6cQoxKx9XtfnQDXt1nkJk9`mS2h0r#@^{nW4Z4Hj%i=>8^8X=PNmM=$rFcWc;BR5!*Wc5Yr@an8OA&EG9q_bh}7mNEV$E|n#}G%=L~haGAS#b%cx9mDwW`@eRe5(Kop<K28b+|I?%WJW5Ym&iPS-fbI&=WckLyu)Ef%}Z&QxE<9n0zCrMtf78I{E6oPX>fgimZbrW&CQ{GRiq3v3kIDx<E6gt54L2=LP_6<)<sq&`HIaxO;^=V?oASQUHOTK8x+&b`h7K%@zL@}xfwY?+E1I=SR>=kT$uf{D81}U0Pg&1SG004iSkX|pPevhvKKchX!ajKp7e&E`%A_+VhA>i9$?^mRKIZvOjB5ZN3wrjfH=hBjS4E4a3m(Wb{{)3o|4^{MY9J@cIpF`m0a6ZfWtBqW9cr)U&$`#B-58&9C<2jXLYJs1+U|MR3NQOhQVayo|{Sv?yfM?NdUvF+Gj}Uwva2qZ<*ZQIG?!J&Q-~}3*ZgP<W-%_qRS3uFL0P45~h56^Uh9ns(8|D~KUIGwC?2uija(P?(1Vnw9@v(&FEIcsr^T%P0z__x{@~93CL{BkhjA1Bom3H@V^vO8D?SfnG4o=b_8$;MK&``HK{hI*Kp3lo9$_RUc<QhmHQ4pYoU9{9y{+6+zK<4b!E`2Ls?m*W#wb_K}SVi3sf`&;Owhj383`N;=jbOPLLk|*|A0xN-=%mH9zATeK2?!YAeGR_!7Ki8mV<g*hO5OZqqkx^^u!-xh{*Ha?b38t;LhT6sw07qcdT&pg`MdY_5XGUwv31JjyM16yNVaJVN4O?YJu7<G!joqK^$D4}qES`tXnqXwEQi<)Q)D`HXY;Zq++9sud&}Fo7Desqd4P9{R@Ni*UWm*NehbBWt-B42b$286pPT%r)du1orSDIA_{iu;o4Q0Se0^GA=6LIPm#~_MCH{%j3OmZGd>e^qL97}}f%J(p8TcU2x3EiI(y*Jl1w4p%51LZqL57GR{J~*^X@g@bRc*=HimwB2`c2JuGwGq(;hrxj;{E2zH}wYm=s}8D^nn(04)F{Au)|~~GEXP(n<5xK@NbPbKzM#-+`MlL!nz%8TZxeR0wdQLr*qc4Vb7%e{?ApSf|!@Gm?05lT^|2Nh4I7yJ4#osu4RS*QQgu86PmBzMd6lSKiuPI5<y|Ra6!v8Jx(Cpc+IOhy9-z$=Z(C#yL^SUin6#iO48U>CF-C+Y3$v21S*K)A`KfwmQ}@wV+9V2;~rI*s{6gtpAz@<jSfqhs$i1%3^2I5>RsSipj+<ITI>$La7@mR_&?2Q`{V(nl|i4i9x>40n?@!iSn^#Mabv;=K7@T~jGCz^H{qm1jnjGsxLYsAxgtX4Fi#B>`&Tm)CqQN!sxK=!Z4?o`!1=rbPhvNb2&^8Wu{{vJ9NovhqHFbqtgCKIp<^qOa8EjQqaW+$`*C{xWjg}U(b<g^W85u!Csnh^6?1B2xtSnRM*e7AW7YUtDcz^yS5~z>l4cswG+ZO}04IHI2i4!wr!nv;VXh*IV>7bNWK#o%Z{P(xqv#>tZ*f3FOVLJ=vMDRJXVHNPJW!UUU_@HJ>Me`~Ecx~8&-}1o1oE*bd^YTMMJRy{_b@Dr#zCIKhFO_>8rg?p4DyaNZ`3=q55GFNAv3`At9(TlPa|xN07QjLp3E|vW~R+`RaAp}OJ74_Sz5q-GH*bw#bU)15Ci)GjK3XwCXA-c(7Gc83ug{S^4byx<%@^1%`S8<n_Kw4Y#ySX=;7;GVPdNz^5JvzEP&)W@CItL1as64?fizcp?5yIl?#K@^5hJ9T5*}2QoIJ($A~F%F`K$Z)W>NOwUKdK+<pn;g^35NbP5Ac(U&ddnUTtbUh3Sw(Dg>H5i_0N9*XG3EdM^*G`e+JH0BglrYkKcCZk|<px4p-<Q0CroVJlLn1VbYZU`8U!JI&u?ju5njBfVc+*k|C9Yi3UC(ap&`XS3URv&pq20!*PRvWhm931gWH84S(zWfX3GFkkf3$5Q>k&^4T=SDU6pdTeVTe4ke<A69d>)2_1u(aq9IdMVYdP(J1sEAjN`$rAWIr;-o{G{IKEInDInCZNrH1>5Kxv;^CZj$DH(Ua9A(~bZ5cBsMR35C?Lf{!tr`&(gc`xy${XdK!Ce%^;P+H{d<!p$MRJjZERD9wXq(bp*GUIMs@YhP*=Z~y2z^N4mNjL0#ikICbM9K|st9u|s9=u{f%J7&-f!#GL)dcz(T7{K9YDPstG%%8%%5^tCN{b(xRSIy=`-d8f>=2iGZkCV6;z%@uK)@b$yj22dQB4|yVlYJFtPrYie0T7W;Xu0stEX-iCpGBnB;5QB^lkO4#u!ec<k`KisXJR%I8h{ZXDK&p>!CSvLl??+S1R1>gs6b{*E&#NQ+4@(H#t1Q*B_w-DpS(?Z_0l$MJ8;U?)QuwMk?aM^SOZ)3KyBk5mc8pzKR%3Y38(EXz5I_w&74$<no}Djp{4m0U`~3AJ;g65Jix(VUTX>wcxL;b=4~oJ>>6)sF$X;)k<-kQXZ~@n3+F0Hm6$YgKib{;E4k+Sag8(%`JbLg%@ed;gjP8CFi&r%In}4<Gb(4tf<Tj+BdX2>e;rC}BvFad)@Zab1V!@RZqWBfqMEK$KWZTzDieDi+*~`ATI>u~@OR28Zc@o}656g+TXL`p^0O4I_{UZkQVpa`_wAp7z#-;lgi-_kCj5Um-HeR0-O{&*CcxY->LU)h*ddbs1a&S2;DyTL+hgivYf~PSkK@5qAE0Lt4(2L`CMX&hVKIgrX;4|b`h{4!*AZF^>HrrSdy;~}xlTA-EqqJ68yM0KqV;r?sh*?LQ=j{SrB@?%QZd+Z5aM0WR09%=Ka48CDJqxzq*Qb07c)y>if&TO$oNZ^V#6bUPKYg8ExZWUgdMLn3gJ{LFTLd1>y9=*OZNKWp{=C{RK3uRY^4uQ{Y1eVC&f!8-{>&nj5zEh;vwBt*f*J}vkB(H$19sw(94ZQBYqWBqb3y^_kNo4Lc1fH{!K@R9e7faV^I-na;4UE{{TvB_jSa~QDo4V1!0NY<KQk{QElz4%>JASZ(RF4hbAtKVPEn;UkJ(_4ZnK%6ytbq-9eUi_*rGvnmJ$RXvWYG4Be1<r0zLSl*;*nCr1Wo1J&DAb5a(v(SVRc+H;OE+9?5_{-mB&cP#t@GLxta?-9AoN2h{y*inajqqwS}s)LGlaE^DqnS^v%74EYt{Ak>!zgq3D<&iD6yz|qE(;(Rl+T^_)rw6+sx1L^y@lxHLVV=$s%E)`P$!HfU=##ou#2W#3VfalcQEUVcg?A2Jxq*ioX|dfRlpVdmu3H%&o)h8r4Q!zcZ9p+5#YJT~nyntc=@&WYCEj_wSa?3wuio>o!Q5}*<j2nV8r9MeS6Q%W7uufwwfI7O5T)$aaUE0lxww_OXHohCDar$vFTv>EnaQa$UMjAXkUQ8{0RB4|j=!38Gcu&^!C%Z+Onw6-N`axav{nnyY-Ho*TRAS@fw&@02?BBPi08Jde%Hgl0rLmubWa5F8;XMqRH`1YR!EizXQMWZ#7HsoRyd3~-PCoYysA!Y2OhnP2Saz33#Y)g-MAP%f$jLmwInLnc8RCq=p>g!DoEybaatoqA=y`G^7Gpwtxfy?gPGQOy$t4#SAoR=j*``%MV+XVM`%hOr}DaLP>03sAFl3m#KqO#?NkD%-&h^G9^24Bd`#ilz5;3pLdpMQGZz>1usj%o8h7l55)q_m#A{2NKg;@%`drsE-~Q_@El8JOU-Q<aR%fQNB+%q^V7LgZpCW{T(Znh6>cO@qiujeGD8mK^--qd26*!-jPbsjZeYaaZ`{c{43WB~^>aqU)n=ZJ1uOd78OG)DqUMLZyST+DZpcgEHh$sfJflQ3jJ}NF%Knaz`0O%D|GODr#La$pgi#dk6=*Irtjg+^Oa^t)-Ia*1tEzOwI$uDKX{*i|2EaAJm7!v5!hnmI^rZ;*U$m$I9KCM97k;W7PeeCtwO4x>&sbsmFx+qK<yyt7ac)_xkS9t{l!Qw%&2)`Wo#8R_NkvRy|a@{lu^oBuy6;SrjH<ROTjNAkAeNo(tYf|t1-PFH#GO*Km_GCICrk%~{pwtY7Cy#nyuw>}R??8y=sB#80@OkF!><NuTW2*Q#ERz_x!N)s!qQAb%7u!8gtY@yR;Vfa6DuB+*ZXvj0<VMl*OyLV%5Iazj!7cuWiZFs<Z8YKgy<-Po$q?s}0vrwO3qU|Xme@=W_ZaUmA=#Jbz*P%ANzbE_Sk(_peJ`fCk`dv5enxmS)sj@i>1EJfU;f01U|a5?oR7*`?(cp_41DV6mNCdB&dJQBKM)izbE|UBPUBd<jom5c;y}4vz06VFuG$ey%iIbWxH0(*&-ZhePm@nwKZlp8W>uc#iW_iileK)ZCY5v7_J%7H>$T72(QUyUOv7&0G{Qn2+GI0Ru!plm`xtXv;4rHa?S~m_kB^pD);&}_&`F<WeADnl$+@gbCvEJ9YMYsMCi=l@e&e2HgT>%B+BU^ONDd0&Vh5O|r)0r$apozkC(Ib<0Y&sz&)7CiMI|0K)auO~Q+E_7U1&kQvB@e1p`YtZf8;Ai^rV<b`}>pr?og7vRzf1r8>$1jh^h?biFaAeVdhj9<=HO`-?v9eIO${#Z6mN#FH|*dCtlgDg+wwZvEMzMJQ`bsLOGqN17D6-9_Niv|E6mCp&pplfEV@Vh~LLWp!|Tp%1>9+h_+!ux3hoEbWmrpzH4-9;+Tq90^eaI@+hEkIMQ#Qpil)@YZ7Hb{4N#JkTqE)OGl6;#>=_hS32yVO$ddgZ%*vM(UrW%?LQyCH8N2Rqqj_JE6Ho+L^$1c$N&u4_N~bE;`6-4PKG@Z2jQn#XDg$sxaz@OjEy=!niC%K7Ph5)n2KdU97anQ$7z-tmMA`9B3IzJrcjvRl~lT><JQl6CEQ8y>HDoVH^OjQ@Q}2Z5nz1H9`NYGYkuiU<Y|Cf=c8~NH<eY+B-y=P=6Ik&=dG_TL!dTc7mK*<@{kP`afAH|Vq>YHih+3U25P~YDm}!lfma=RW0x^BM=1*~sx-U+>;5R+0il&_+|I(cwoi#md4_wPq6CTV>6NJA!aM{F{ZsqZ@Xl9)-CmJe_z;K2FKtV(Z73iE6;W%2B(blnAPHksfLKdhpwe46_dKR6-NeUPm%~k?5i+)94EBNU%-MDMTTv6M^MEm2#aRUdtIXsl`fW@?$E3glf-TyBwE&`vA)Qd9IM@Vh;vboB@i=cyelVv%K894rOs)w@Q_-9JqC~?A_od{)M;iSC;u^Q!DU5NaXE&ysiL<J3yD4<lXE<Zd{dw2CNlAf{1}p4uWpxgMTpCj%O&Sr~;pj{x559q{{{DcaLtGP!vx-9Fvs#8P)Y4JNG}`R;AsZOUJ;OBnWJ^?}qAjRxvQKD_TFFhnW!IT#MHWM)XiO}EYs_t!5!$AOFXsTK2a44G*k>Oec)e&X_cEiWy!OM^F`J<pspoykvVD!Hgdu&Re6{=EHIuxj+*2wO1fPzpDsz;>K9cUwkcE4&>cbQU;BFnzs;8c<a$rk^oqXwe&$z`k?Qj1-0QoX~Zw!nEGo4$xVAB`O{tG$ezK_+F2vfxBucbPAO0uC_*X^r!(C0aa7l~UrY{n@KSqmU=mB61quNXf;xNLTIyOL#Gas+^qs2Zjs^R)&PE|E&`462kG<HHWv2-~ZOddwSW`YZag0aF0a?iYZ_fG8%Vv4M8B9)`f;l^koRqzmwBjqv#4e&Levp8EX3`XJREwu7wp9|cea3bx*12d571j1Eg@Pn<WJj`T^sbkpP<6KnxbZEaZHmXD4heMHinzadi%A>PifaX>?(%NYSa<cwIJzrR9VRr9CPSZBnBn<1B3dX~na(Z8NaFys%`IM*uoP=a;cABLEhaVVZowq^vgHKy%w+Yk{)Zt8s|V?-}8`3+w`?BAE+L%pC$9|mqCv7!pn18U55!Jrwfk1vV>46(v<*6&i8hW!GR8d`}>=x4qVwq7qNd%m!BXH{dHbsdCUiKm=8$HBCn3|{S3%!z(c<5SIP<dAH(Zxf-#IqDLm0f9=xVR<RTNOCT76D}x#Hf8uuvh1PhJ|lkSHwhW02VtpWD1y;2bvFBY|6okJ$P(ksJX~QmE?hWSpu1L8h9x_ngf-QD7K02(tu;Keq?G$z?j`-shlEE&icl-~XN!poG8F0BY3?kar)g78Qcvbvrh%TjH}=D@4)(PVGX#zzdLh1jC;--WbT&Is1|_pKtRSl$n2eqB8_1{toLHfu8Ul7yFN_*QK~m}1gB&42TBRnx>sM|WjI=6?0SS6lP<C?qP46nx#m2I#PG$haZ9QLIUij3ME5mu4*(796O9X<^_@X1BKO~l&H-@je&Hpem4aSb6VSG;Lp3Fd5A{<EFBc$FmF!HM6Wa3Co{JgYn*0Mc(ra#)MZW90oUqyW4?{<B>=n&4mjaa%pQzA6lk4nnC7mHOkW29{csIbO#nF77nhe>t-R;*wvFnXfQQBJFxPej0gg#~PfmZF8~Xf7q_<P=&W==L?DA@dW`Uqw}O$uAT#x@=R}50%?u9`8lfcm*G*PH)sbuBc9qvS3heD1`59gjYD{$ZU(dpCF>+w?KQ+*%^Ep7RA5=*k{c+OXL_w$sX3n$4s;UsW4Oz-W~KtIsnUecd8<<=h$`Ltt&*i!7>Y_@2)zi=)=8U-}SO}X5<$pH9d;G!Rq!o*R#b9ND=+-Eq`_&wOdg**mFwWlpFo!0qRw|aNu47=mNfU6<zy`-B445;%#Fcphs`J4U)ZT4&U7ikTY2`3gHv7g)}El118vJ1G*7DAa?Yv4wr#}n<M{`VnFv9StOw273;_ZwGcZMosbwNNVz@fE(^hBy)~RdNbEh7UCKcb0<QotC64N;KS1~pA)>sk@tfg~n2l!?-muLyQcZese;jU!8K5e09I@qwd4*eT!!~TGEKK8rl3mg=l%c(ek2X7qTe4aCgWvtsPkJo!@yl)sN_<BzY|)?yKj1QX`F<L@Ms!F7-`r6J<ZB6k`KC49C?hmOk+x0oupArNvMadlB~K@?_`uc@u@47V-97>3cLjs9Vh2Pyk`v!a;0M4ibS}2T8l6{V=rNc#vp$rf_v#f(fE!lt1t*XEuFr};f^t=^G=_ley|t|~uDwxOgmK`lGV0KAEtHq5Ezc7t&s3TMh|0mTf8Fr4SCTyTK`SvJM}9yg@Pspl(LGMq-d)b!uQPd%$RDJ4jR-ou7~C-l=ATE8gB0mU&>A9k$7Z+?G=`uixHEvA3BWn_4Cs*%Z9*ogMV$+&a;AFB6!0W2-kp3d4{}+-Oy_JYgHHaZyi;fNHX~KP1j4hI^<u~eULb6e+|}wzs2<fWV27u^yxJU`OF$hNIRi<oHNo=tX1sfm%O@cyanDrtl)|+3%@K@Saa}*&<-3_sIHxv9<b#vxG?b|8|6x-x4_Pf_!V_OSs<_c2&<<=26c+A!GcbmS@*^m}+%M_$Z!knRGifLWn|@INd&5q1%;CGw+L2ztkN5-NARaVN-xl#7CnIK~Ym*|bidK}FvGW_VXuGt<mLOy0%o)anp?W^bACk^0fevW&&meRN_OlZT@&^2x&{9J>$Oipj<<z$giu&#VNA;pm2k+KftC&F7m`t3^qQBw9@vch*`_E_)<%m8oRqpsELd41;M&Pue%pEd+m9@K(<lzN=bUBX!b-g)lZglITiYz*~5vj3LTzHjX%Li0{%)@-=8QH46Q+cUxw@fFKdY8s_QY-3F2A%`bgbN1R9H#Y*#Q{*#4zq_h2@EGk{^0kDeBdyY=vn1jM*|azXOz#KJSmK1X;B`~hH*-A@<!q-Si7-emnCd&`yOSjKu}0*&&SK#=Iukt5}AlW;2*z?Lx2GxEMJ=&+}K}NH-Gv#2TQMtYFAg^vicT&CC_$_cQs=UaX%KcVIcSf0)W$xPGNy^kY>&};ec}ZZtQH@RDwaBu9FqU_^C9R5J;<-)Kz5<x!<Xvwj`+v$l?z^7FbQUe`8O;BLjRuY*p2kAqRP}bjJmXUu+}^3K<&$Vdbb8@TfMg!y)hs2f{;<N}>|wk-x7E9KX89pQbWO)`zPEz#R_B;Bfk|*7k9DT-w-M?61|gOsn!&D@1r3YnSw>Ej>}o#xp7Wyws^4dSCJJhsZP%-t1KevIB=-^)-YEwkT4?e;)FgxBJ))T%!Xd;5Gy$5C>=^2&``#hqe;-hu`w0B5v_Rps30Xbz94bRa{ydsF=(1?buRmwwAYfEt$t5#kR(D|4J%Yo#Tie)vZalyiMyosTpdh=v-$;{8@j~X(R7wBAlW)CBd(%ksZ$HQgeV69P_B-EjK`t3D&(*F^aR7xDq0m3MxB4PU#ChAER=f0srJa^8g`=1yRE-@salgbf`xP)N^zqbXe99JnJC#=r(lUXam1VQaOhA@gmk4%@wUC=6Y_Iu-^xoia_))nDz=d#L9(&q_3;75eOorHcQ}w+P~1n-K{5O`qiF)gSZpMBh~%>Xnj1<%v=nkE`tz5AmbP$I!#4CsvjOtn$bx>!;iOHi@<WDuhoM@IBEl*0;!g&D4%jY5zBl{^DoJp+<pCktkaM&{Ac;{=mVJ5bco47-^za$Q735@Qc`(0+Mf4<h+j{?_x+M^6~R6K=$6wYI4-`SOY}eN(2BU=Ga}gs8@frOW_*s4=l+LU;p*5~)4yT9Q#`QzV_>j`UmB-H-7vHDP!2E(dzQxc)_ELBAF)qD_HB`aSE62QFT?SRs#l(xobO%@j!4izLq2bJwf0KIMlrt|;6TyAM5wvW+TNQ!*<-TdynY48>&xfC%oS&w(6H2DEZE_(KS~H59(!*EFh0R3g!u4KdQ)6`_9<EsH+Lv5XeEIgm_{fwX}C|QIn>yWvs&YK#y(KoF*EHzv^3mwDQRHY5e*u<PqEu1TU4lG6INkyrT$Y<=Jp?vl^0q|<7<uWNf!;u-W!}7D$}xfJ%S9KB<%{zwjbblWW&3y*4BWBrAX>#Vwyo|dPEg~s)cH9TDIJUOsS7{UQCs{M#B*r5QbabI^GQkNo+SC+ZkquL%y6p)t#bj#=(|bfHH#NTTm7`+8fZdTmQvy?l>VwiKcqRlI_%?B<N!V2N4AqZOSx;nzae(!ifdVfSB9y<CJhRm0XK!H=_><lnPSv_q$#OSYRC2-Q_N<vy4wnUe-_X7D8kVQGM$ss(VG$iQtSS>{l`<x486`&sam9(eG$wGxn+v<afgKUr^#G28M|Zx6<c9hHo!ZL+~)3q8{%Mh4RY19i5E)i(aHc(1M<m2M14%tC`yZu+XPoiZq<a8qj8VCFMMpoBDcq=A!tcMyxV6NXwlVPr`5b68v(1KIV$Xc+|{fN@!hD@!mK|yaVM}>X9mhK;DpJ(A;833un*e&NfL$9G5#kkB1!4AnRN|XgC=|Aq+S8)qp@BXx|g$BU(6W$O5=$#2i-<0j%Bil7GZc?G_Ov>YzzX)7t`y1m(ia$pmdgTfmc?PQ7AnaG8|o7K<Z5-N0kUYUXNAF}CJ6VshVT%O%}R&9u2&7kPV{4$#SmQ}{8SvqX+pdm_L~?ytRIB=bX*5<FF&bfvS~hGm(J$~;1z?Aq<r&{XfQCeM@jasK{zH4IFc4Nn{qKySwA)Z5I28_%m3);!Z8>RE)J=Tf=XyTh#cvN36=`xP&;dqxr&q)9Y+f}q0~H^JA@L~o1Z5C1f5$pn5m*4=c&tT4jTXml6o7~ArBFxg$wAbWfy0)A9X@*UWXKY@)V#PI3U&G-{%d#51GouGr9Y@7?k?E0<!nDEk@!BH{M)Bs%2^HwxD+K(HKHUKzWX_@3MpuGxqdKDIDu);0#;4TRK6#B9*zQUKDGj@YjnDe~<QPSpNQk~L?kO}Xemcx@Pl8O^Ph5)5s2g3~m3j;Cng}Nk6Un7Ag6w8cD`Tp@6<A(`%C-E_Dx#iHDMQekf6T6si`Og_Gtyv{Avr11~lPCN&3yo`|XZ}+(nBtedA)-{*CS%?9H(^h<og3=i()nG!8DK{V<Nu~GH5zNH53oQ?JFqx>1Bwkda=Oi2wjckXj%~0|u)oUhre|T3+h4A9>ubrB!rfZ$1ADhvc1e;u)UDQDaY$8M45l^@_6*8fwguxcx>Q;YJhvUh{B;PfvQA=ONx4)^eS5Ym<0Sw`=>@7l*%2SyufHcnHy3W+a!htf9vM4mM1gSRKE~@JY{{y9y<==4&GqF})M^}^c1F=;C!=Rm35rIi-ZCaqWR;TOUQaG;d6Wl|Up1{Qdjd<Mlv#Q><t>aEbgCqC;6OGynu;VTSxe=aGSM4g4MqwoHp92>#aT*;3-qntSM5C+y00Z^xjNf<R^vQKX*`DKeU$?WLyj=K;JFMim*mKZ^o(_khmZM-8@HaYHnAmUS67(g&-539%tKYUmOw+v>GlPQNG&J}1rFV7h!C^``TJ9TW!+H!9JuV6ELpa{Nuysk3wU;mkcQbc<K|W&Z7dI*XFoHy*&CVAJ6C=jq7K?XKsI_4k%Xy)%a?BFzQH!$e^(3L`Zm-C{KdnBwcHq+<*VJOYUgoOwtTMH089`|Nw64A9RX_9G>W_1u`e3oJwPkjxlfzMhx3b!Z*ODJSMb*^Uqi+_)^VBr4G7^z2jZ~4Xu}xrlXKt{h1&^z`A~5@w%_9!Qrhde+Ctw&nX2KOGaUlQeHa9zHW?>T;s8D779w+AH479fU$z)Ch}x*Zv>bV=3`FV}ENtf{`&zmS$ti&9hCr9axB=6}#+d-`ONVk@R~lMYJJAPx*D<V=Ok|!4{DSf6_W5JSMKht&Dk&gqTW|$^HD`s;BKSn~-F8}U#cFGO6LT$w-wiV9r<bMJof+?MJ);fk)ohbT4c_dtp}8fI3&5&S;93BDxhJD1#N|AO?IVH$!Yg!E?>|c;58e?CuiVg0&WSQ*#P;4@{$#^;UquxyMN2COef;b{8kUI==|v~l><6+%H2m|rqP52FK!um|GkFzQn-OO*WqiM(-&cDzDfR1Yu5uN2I5&5!=y|D6D|$HJ3o%XJl-1*Ui2h=bbfnqAYt}40T@ew^zCjMmejYx`T88c1)6ll}!NOGve#0HGtG-R2ZqP3)Wngr+B<-`j*^glLH8bY7*LT)2JNO-6vC9_$?n9)xr*LpOuc3w5=VO}WRXaJR{ctHUYTed|U5sj`vmRpXEq4SD=q(PA%U;)iRPe>IL(^>i3$esMtzR)GaD7}Sv*DI^i*VqftzSw@ghb3rmS{e>P`_g5txvh$Hj8yS7UJBArA$rT!0E)dh=EN}OAW?ueVyL3FGoO8kwmAED>vM`VP3fSZLQdDdHLVQU!=E2>N9MOy}W|F+nkK69VAMz+!r1&w|98(tnm58zvYxGzssh;SP&T0xm{EqQ2Nu#8)g#3RE?;<_tuAdWeXF&Ndv!);JU&wN+xX$McG3Bi4mtRh$JhcOp-L>1AS#Js@J)7_x~-!#2(~P+r6qeqk7NT@sdU>N=)zA=bj52YW>5ji*myE1d~ma`RfQT@1u2*6(SvOl$|szsLUGLy$J)>Dr(5%{QS38fIA>ZV)6We)`nBu$zuXBu4mtFXeunFh}Gh~(%mP{&NSe|S8aI>wn<^!wehJjc+q0}TEvWywa}1+dj6{_pilgL5F;_>FVY9eYpA){C2j1*ru6{#&_3E{Bd4rl(7Dmm{6J734Nels8Wugj3yTS%W?7={N(kA=)+`;Jh60~KA<52{@V$-!aJTMv*U5K7j&V2;ML)fUG$gy|osic@Iw$^8oky$<pVO6rpiY$+pH@4+U^Lto?gb^OB?}AbIrl=X51R60LyTr!<3?X5iYBboL8ZX=^V36unJiQ{`h+zM>H6<hwmaKJIIEYq+jB#KDjhf;@xOI0)-C`x&zXFQS71p&QydhLA+%R*&!x?X?cvDYLga*YDejdVw;jf4?90`Nk1Z>dd_f=^W&=a(H7BjK>_`>Y1f5-Uwnjp5nJFOe`SeiVrQDtAGw1<~CaOutZ9iGcK%*gty|FyDBbx07DFL+eUJ|5`K9Kzsa95i+RV9wc_FsrDZ?GTY(PAg~M;MHABT%(K#H~aLqKuD)R9o}R5)p#0_<m6P7%&Qo{gXShFC7ORcEiwEIMNk;R51bxMSKRigN^$hCLeS~M`!6=pfhBri}@_NsROp7J8Hn2a{;wt8rmz#@cO<6tL4D?Q`rs$2!`~B^o?TPN?0BI!t8S$P&>`caU;%d-Ns&6_dynY_d&q8Dv$-sz|YIFS`NY`n}4o=vgX=r<@GMZm0PO7s8VO)DW;^CS192sXhP>Hgp9kDH%q-lCriO8=W9|jcldYMR&*Fl@_wB-p0kwNY>l<{{<&_*K2m&=$j1P6$s%NXJJ2{)+-yC(5|3?OuH;3d;d_c&M*Gzcg`#s#11KKHKm?)~79O4`iQEG{j#pRWWMnkn%fg6^ad}EhKj)X|tx(VkTaDNVf8`Sbk|_JC&0ts7@c-euw(k?>TXc**p%)E>GM8wDP3T}VnyM!$ZbALSO)E95j1Z^$JFv^0Y3i4Pwj+ZkXX#eX!TT!T#%`1JY2{mTQL6ld38ay-IuLX*P}^y{c&S^%ZBw*T4&u(sj)DSR5n`Z7USg`kvwkDe5~NYKRqS_;=;*S@s{5E8$cDzDm6;TVG-_5L0Hg^<i<vbcrfJVq!U{y#w#PHav>(N-E!o)$8f~e;&>(s8;M<@reDWVQFRBf5p@Kh)I-6pQaS89;&clEU8`ZG>YgR$oVD~@uB#zuRM@zx&qInb$37R(FQq}=OFy;H}yJjomCC`GJWYeLv@{(xK&RjsMoZ_}rU1qC5Zm2i)9;wg*aF%q|!@0#Pq1>wmuq$^q=IW*zwB;2DcJwO#Ug&5o1}c)Qq$u~YpZJZD-~r=9nhCrYyWHS0J5QJs3CTk_Sc@nIH`tDTcYPK1zv_8ymwk7E$*I8(PYbHQ<O-0@(07%5FiNiBOAN*Cokwd;*U*x1>*l1yT~pg~BPcJT;$fd)IPM246y>N82n;~@V3Dh4Q9KwX_xq)0PKo$C`{D!?-+_jMvf4PNhMHbY5LA(MN8APm^`<m$N_*Z|$Qnkez1ECrJq_A+GAF4=3;4awNyTf{uP4Ct=ARg#{D^sOM3Jb2RQ6typW$X|a;FLdjaNRiau_C?vhL7+C=R)i+p2i%zFQ*VAbA!-Z{?(0y<kuql*)X)Exq6zBPY*j{kt#^`5z0Om__{1ApYH%0kL5^hq&Z{itiJEC9Df(=G8m3+ej8$PLcHeSaepTmxgaNgO4b(6D+IbbLZOHzXQAcKP(ouVFB1uI$+w|`o4AJJ^X}!!#K}s=#6(R;i2a7RgVQOK03lUC&2p4eVMF1fljG3i7c(dmEx&8=#N&JIN7T2a6H*<-=Kq}-h0d`M>k0tI(M+v!;qC>p`{K)*kc!{-`D!9#}<E3;S9&*BnGw(`|wwE1rPD$HLxBfE%emjbT?0-XF{W~pJ+z6N#yR!!7de*YGEI2el+7hZE9fut5YzpMaXZ?X>PD(p{EuUp&fX(7w#lcH#Ys*|7V;P<1!LhmxRkS6G_x2j6y^lV1XrvbER92avj&P+pPIC9N(j0Ap3B{*}E~cX-JFDt;*3D+;Lt-8dKhcR<}R4_QXUhiz{@m#v|szbq(Bx*<$#Z0sQ+t&*n!JQVx-}8sH#RZuBZyvHS|GT8aRW*fo+S+VfoUu0y>H3Z_5;0Wh>H%$VKSXCEqp(XOP>$TNMgs%5Hg#hA~fm_`!8&Wy@|)!Lo<wEyfc5#t%#etpJ+P0r8k&ReQy3YAxf-eoi^x4=i2+ok?J?YcupGP<=(H@aq!3BqnTLki?#o*&MOm>*N*w$9%SZh^+d3}FhzBwf?&zZY~(td0#_JYBz*nbZ$O*k(Kk#SpcR|MZhG;3VntDSa6_9&l16t+FO7i(6CH2a7B>9Y0@CX1-D&qnnY0b|Fbe1N9s=t15;$17R9OU5g?1r55A0toIa}6<Sx4ao)J|p2Kh|UNNKV!ms8TJ)Q>x20%fB!+DyR#I&JF6Qszknq!BgG^@FQp)!r!nsiG$B08YtMB1i3LP1HIadMyI)LAWs(og5urRnR0)-^5II^(URAKHVW<rRy*|6p3LFAVUI=~dIOLmW>?zQt)fA@ltP2fy1~*t0#<O@Q97d0`VDA)7VS2P=HpYB+^u1F0Gu!H$}g`QrF!Bjuy!Z;dK$nlhhk+w6`Iwv7_a@f|;MJO;g3xh4pw&N<+&U09VdE(SiZrB<Z*Qkr!2eY<L(n-l9)fZF=#a+Sq4c6IRuu^+Z>v*@jm5T>f8t{hrNjWbP~kL_`NKaU(3v?|Fkz)^V)C8`X{k%Yl_vNkmBlW>+Y4n1)2(x%>R9w_VoPXC}3uJg;`Pvitn?Jm*WC^q0CaIwsSZx`O`xF2O0QktAOb=%?QT%uo^*H<B@d_f>8<^!UetnGP<sLSE~(OIcGYePtTS}b%N%z4T4syIlAeB|5v=~s!uGr<8f3+NrAx8I5O&$4$Nsv)2+a(2`SnECCJ3BMgUI{{uZyVSangsZ#g7~f&IsNHk?rsIxAG*V}m54bju>a>7_TLd?*2*|2+L`~Seug1_r3G(zp{)}!xBn1SOgq)K!V{jj6h{_B_<)qb+y%MI;9NL&dQ=-ILDY>M;1TCRfl+<)0x_E5I<8WLQj!K!wVZvbM(wf*5xERrb8FC!dDnaNwzfyo1<LJUt7cGOrT~p70ga%frF6D~vaEp74Q<lElU~9~`o^NC_x+MQBP!m0b(X&kU!iT#&y$r<dudY^N0gs3_-iu^n)RJU3$M@;*hq-a~Xcfk|Xp+RZnQ_-|Zb1H)NL<npSbp>5Es7X%otZd`<o!hI-Iv2W!ACs>XcX43FYZcgftqH8d&AM{7Br3n?WZPXMxEfF%k<uiexx++;6x<@#MrxPwor@zjtaUj(<@0Y#kAd)JGa^UO0faG)ZNI=0OGp9KKd<O9*XcjC{2fRh0JsnKG5b=?sav>Y{s_*l8Bi2;;~qMrkMr*Qp;9{q<I3tpC>oz{wAA0pr#VxkE;@vkg79Fj^-@F5#$sAV4b!l3Eq>`bWdnwLZ&k{cjRqWib^rx!2>dU{{Tco`*l#=<<uul3VV>uU}gfAoADTZjrJov)!mXnjAjixhJL|LW6E-KE#_OYmw%*wjyZ8&^Y-(h<eqMkXpOLpBUpB|KV?g5XB_lY(53&G$1=Ko^v5gIXy`Z^hTdgga*g#HDq`w7(27%NM0aI0$O-yu+x>X*5!568LmS1j`;w}__RQ9@#B@F%@lWVrQ37_CUbWhEL-2Ml>mLR0vW;sM1jpN+wxn$tYWmsa0kBHBUw53u&dnF(7Py@tRCrRN0A|>%z_aW&N8Z8Z*|vjS#o;u294+q&U_LdrL(Q<SEx}tYvnNRoC!?8F_(v8?A!F~v_Ep>M3!|p)kqykI1nRS?85<2?PeZ&=XIVemR;<BO0R=bXwEwDmksR1!i$OoOiOr<9=f`008iv!{hDnUd?OT8Rm`tiG?u&{nNd|sJu-5ETTtDYU5TWJ%(|t)f6_v<qlX&PW3!st==ewCko6FSbj&uv4`@@KZ-pQ_2+`yD+AK&2tZIFClDB`105kp_bxJkWgqjr7b(A1#tMK;0<l4?h>r2+B*_<yN&PqcPkNXC6r13(a#ZRrvB#Er0lAR<Ua>Q<1wd1YUH++r?iozpI@ij@OMgiQuWPO#xhOX=f@U#WHR-k-SD4yyO--+S3Zel)oSLu&K4W)=WNw5bj_`7E9~GGDYTW3A97_q*NW8u!zNUiN)C#c9<1agSP9UR}fj0bRl`1~79dgS0GOT#BH*IZRLj+fx(+Y&IXt$eB%@)OXsW&KWZdJEIoUC~XN+J8hQXeuu>cRiIIu4{NVcUKy&mwh$?p36YDMdh-~);%NGs^o5BlIwxT2;E`z;3+C*eH&?IPv?!U<0+W<EqK1M1W2x^D&NN%W5FXk@ZJR2$(|-$lG+ze_&X&<BuqHSvZb#Y*@pu-iAKSA8Z<f=d6!|7;AX!2M>HeY%<`4%b@5NyDpX~v7R#(mEYh)CMP*BW@sv!RoJEYha%lQO`27d>zf$G3|+grZa<m?ql_9MiQ=JQ&nJV3QF_G6wy@k$oI@}t2;rVphuFx~EIQDs}DJhQ~!)8blq2;z?W;}J7=tOgso>7fw*cBGkXCQN$6()tEL`R(`3TsCWenwkwd5n&SYw=?qujFWdQqoxv?H;6Z`MLdkSA<x^6B?~GCCDixWmvya<{rj&s7h`-=QM)3yNZ<?ZDfJ7Zl-zHnh0d`{;ak}<>}vI#1v(7L!2`)(j{3B;YuTyfSkn*+cnMJhG^F7q#)3$j=-^S9I`~%YOHNr>ond#}XaJsC)BOccfOBbNX277uV;bmkEQ|~wBgG1XiFUbu>aHUXhkYX(x0#Nl&AU7+W7Qwu5kFwE3le<&QTG`K(0?6K=t*L!#d3P9@BX&Do5v;|hew@Zp8IXf_?ZN@$-%pUJ0g;bq9_k?;C^Df2qY2bQ5%;~RbA51Y1P@}0qC3|aZkCVXVSyJNF8fTE2Y}&ji<{l)>BNJzj!W$TRLen9m%kSG|^Z<(U8qKUziWMSCwx4*D@CVJP#io(bT>CeAvcv;XCblye5>%`b%Ux-Q*O)+fboDgO&iFYSFK@)NR&9Do_3mku9$%$N|C$Qg??0O>cL>!<H}>J|3ydlqb5D?r|#ct4|^qFg068NBXk29G*Hkt2;NU7URC}-p)Rd(rBTFsxOftGby5U?(rBaEcBwgKD<Raphk$yHTKimFbX9t%fQ=)Ve9-|A~UO7=`jfgpg%1sXJsk-^SoYndu_!kL70P{vSIRsjD^OcO3D3*&?{4c6qXZZN;l%`G_mA;wbV_ZU*s);8@q0Mn4(QuHdU8oJU`j@1x-{RI8<%(u0f<pA8qyYq{~L91}$^dGuj%sLq1ojnhi-XT?gEg4#yBV{MSOTFN(_`@EKPsGw47uNP?9Efj@g!8bA<LRUnl&BX*)xt_`kE8mFwqGNIqvbf2V=tZqH>^`N#+5f72%Ou;2uUL4GdSGEHi;eL%~6!+=IrSimNNCzB{@4ylx9UK7|!gh~2&dvJi1{P~VIaPTb_Le_0>vgNy&7MAGaXk4hl-e}nZHlcR56Mm=S49i)1NPj3DSdHI{IC5Dex$Ol&nyH@oUw@jZX~t!NhQA)_g?ST-_-q~`mFz1$h>)1GxuQy3|@u%b2L>*tYseDvY`i%H;$>Lr|ib6V@&62SHtY$!*80idVrIrO+1En3qq~VZtgq)C_oC66%Q7&n#$sEKGzei;9zwWq9vt{3{sgv8#7d|f*run?B7T=jfnE!QHPo6s}@NQ+<Hu~zeyU}2M0aCEI&M2yvqgUX}uoO{b`)JQTmj58$mpOMEaF?s_s|^TS37wFmLr6M*wsyh}KG#cXMGjm-*cv3yr5=cd?}ST(-4;3)KQ6Tu&f)zWXMZO%gGkthx;P&Amk!4P{-;QIA<G7l~q!RWQA7`lo1b)&FZ2`FT-^2m<1`3N{$1>}V+Y{@Td$$mm807Tb|6ZA4m0Ab`p*@a%)5LYs8Qu`Z5R=6?bQ{<1>DnVmR^ws2!jCMjm;`NBB{o~9R=n%G#d6`qpJht)jo+<vHGVi@i1Fn=;@*SLrjH$p^3{)Q8_B0vu*f5hb*3v*P{a>$*wrz$lc*WPpt^~08fP;q>B$hU}Vv9}~mBSF#!x!XAyB`H3z2IdA)44nnfCm__yNpCI#GZl5qepC#b%yx4(|I7FRZl=Ar<OJ8bCC)Qbw|{866b#50Z?k=#?&R?$Ol`1nQrmR#&GG$6L=sqj_evPtNI^jbT4wT{>~X0-s4YMI)`lsPAAZTYnllT3S-P;adK^I9b=Qt2`dxY`94DmyqA%bVMpIr`(Wn%h#(M79P3PBYU%iz^dx>nJ2<aKy(#vZ}INJnkV&zDz(_I++X$q}ABlDav7Nj@rcLy{fv57fU(lWauismYG{rbn-jeD;}vshMr^c7qviJtOavzbObZ5zmhHK)H1x^fS&m2Ji(a2AOY>XxTQV{9m&hL~`(!litlCT6E0*UCJc)OT4yl+`Z@w7zE7#nM}<)OyNHryn#1J`jPGc$pioY^kHEXE&d23D5}HE<AFrE0i@nk`tqU(1h{qoiHL++*wU)$0`7BDZu^{4Pz4@=5uY4wC+gIXky1DWNCqkR0M(UoQF(#&y{6JD+Q#QAU_`+QT<ftmF?>ahppG9`mg94B1z8u)_wm+AC+A6g9@+zxD*X_NWb2m9RTX*2L$32st6{q>MzEU$_w5ZM72dN?|Ln|pQ8TPLMJOdl%4{A^OXu#Xj(H@L@iY2-~32Qx3bxPilRf-rXtAz#nBgQE9$SC#6(|>2Y@dxH=3b#;OS3uG4|r3yOmg{x-d0wT-Ua#3WN2TB`ZWpR&=Jt`<bHFI^e`Zax|qHn%r<*`Av(29t~5W1Om~lGVw-CRlo4DbhBISf0()9)rL0MK)uMTMJVqs#jeEy<lr$rS4o6ay^RugB-Qg)u&mbOamUS#XxSgMCnA8QaHJ(7jFsZUyOS3S1)R<|lT~;Y$S)enM;h&Qr)Jvu6ztkp{$(5^JXyP%VuEFD%M8+%@fYYxP(gYPwU+bc9P2q65NXofygxzrW-QQCA7##Q>#rBH@gaTomO3W$ez|?ssjXy@ELg@iaw0rW5#$JOzyZ?b75$M?W7D4%DX?Ya&MZZERF<YR$5*d=R{;*dJ$EJki$!hJ=!T7dI$q)z0(aGnDoPowU)uM|NB>8ixW%ey#&Q!K5vw9sI5=y!Oux2jcBSnFKcN^supCbRYPXpeTjqGh6H3KyGOs!3Ei2L_34d%6?$-?hqbwP7C9W9T{ebJ67DJAf`ETPFioScqVFx}_(<wZkP-W`YXgon64bdTDJ|`dHbctaCRiqZcF9`N_+PY2XbBS59At{ncXcLQ*$K7l>VYAz#rygw`bB&M4O2|<Y9+XkeAolYO%@A?oiPlCF`}m-E?l2v${7q<g>8RrfuK*A$WTC)byNiq67&HfCWP((A5WpT#RBSX76t&Ri1R8~A>hs1^#AuyaTHII>MgmCFhpAjrHNf6{z?YXIx`zU(^cdr85^LZ}&sbT7*`<sKJ%F;dvw<~c(pos%-vXH>7>P6Q-FC`xStmx&X~goN*K>|VF0`NU=T#c#YMbXzKPuP-pJ}ThGR4`?;77}nP9ozF#bT@bQR+pRR>>?HH|h)JnV_hE>9p`21FdctYWd)<Yyy6ym6xi*l4v}+ehg^9p%M=S4)*bl{Sr|DxUokNXI(2|SAoVv`xL*hjt`W-QB0?$>KX;^7PKvSpCO00liXO;tD|_H4APQoQM;#N{XbTS%a4i0QUw6_x5J^567>?kZ}G2%)LX_CiwYAqzpf(XcAw%tvO#YVbeGETKK}Gf*05dHIPwU@32xOZhO(=6*k5K_dTyus@cU||J#q_1{bbRJydF<nBFX%vYXfK!8n(?&so2J5RMiLV6tt6;FL9GVX!OoFN-?c6NLW1hF6y1#JJ)T&u~syJrXIAz<FKK+R8Lif?i+kzwAIf4VHdqX(6yPb)uA%zL{tr}4zv!jbB^F393_o+r$JQh4Uel_ZW^gW7SH<xhr*wbMko(b9G5Wd>A666%cJj3yk<ToaU$T(0d19qxC;*y6!wwAsDiPY!$ZG+op^M{+>4~kA>vX*zu|jtrBN>33|eh~Ne;i<FHskd-qCpqi&QfVUN>(=O$pk)B(pfvERHO#N3ZwQ7zfa$bgVm(VlNDRRlvB=F%gF)j3*$%s#dmB6Df^@sXe5i1?78<?g%L%R>n4K<C`|lx=e4+d8qXIv~?HhfY_wykFoHSjeErlrgVE%k49tmbT6voTuKkB3TIJ_;mc{k)*UaQey$y|-;BLO_~h~-AvN&hc*t`O{XYHP;C{G(ib27wHL3cCB@;1&9J;V%Y{%&^88`4gadn2Ec;#VLlPwuK7`F(M2PV38Q3ZCQ*#ELP&lcA=8YfmNl8A&TC6EI+h=MOSYtP<xEeJb-LMfCX=vT=2Nt@!-J&moSDyD2x#dm}=7l0;CoKG*76*FJ*lc9XUx68GiQY=0~ZKBRo8AxqJe8oP$QsgX_&zmj8f#omkHtCQ63yOCOA?anD+#Dtm7k%BH55Jt05WF4AB`u7IT;&(T*UXV9P6=N6+`la_KGt3GeR#e}Mw_W=95*ZO@ujyneJ__Qk7sm6-MC%-9RL@i@aKU7S?3y|z$?(BTRevhsMwF;-KK0vlZzQ<;pq=c+Zgh>2Rf>$g6S@nNt}JP{T{uuL;1QYvvm@TchxA_6)1~V9V3N^8J9F-PwC<EjFDfH8EOqrR=5Mr{BL#q8+L$}B)6bZ+rbRD#8ToMZe1aJRgRSqul4V^4^qjgU))H)jSboKUpfR?ehfLcpPN0Wf<45VV0gYFp^o|GphoXXMdK%l#rl(IMx;PAM=Dj{Qk}<mx=7$9+Sp|j3qxQ-?MTM-L(TlU(m;VM97OJjGE)U3J5LexgzY11&|Q+9pctVr*#V4Wc0H3(>x%^U4}BBfS9q;xkvQQFs`bEg_e$Rj?wlpH*g+y8Aw`uH1g-7gV|%L~*^?kBgy*Ewsp&T-`1UX|b5PMU#i}<k{d2W^q!VO$`yt3>%Ssf6yX9(!BQCzvPQ27=t_Fs;H1P6kCZ8riGfSHs^7&#p%)o`TCV{kfaherC6YSjbl_jU(DHUQTCWuW}zjhEWMvvLx`$0ye5T7w_fV7A^X!R-b3?9ZK4=FwE%2;QPW*CR<yPtTT8mubIo+s$Sb>B9u^B`{+VBIlU8sGRmQ?_OaYp8r`iiiH7xucvWR1?%hu|=nssrIQl(}slj=@DmLT8sXaWrBB#Ul-gQD>CuM&E=C)NdY>AbWzLl%VFpm5Y!WmQLTvn`PpZYQc@}TuW2C_2;}lshlk`RKptM6zYVvf=l=RqtT>@26UV+>d%B)wEfxpKTM}%bmi@b2qZRX5)I(uQJTu(yU~t!$YP{xMeOVnUp3hRCKxqkLRHL9J;62I)8?J;7G)IV1Zc^Kz>v)E7ic3GefID7Z^+08bJejFtkFj=Lc{UBwIo6?KoHlQwytYmdCZGucdf))y6~ZCeh9ri~#aStY2qHR_tLbPdN>sg&5c<Pl<>#i~Z;Ns6rAKU%$@c#vIjydWsT1N<CZ7n%XE8|q%m4~rG^a*VvrF>MlppfnjWK2IY8+UtTWz&IPtcAZ<D0U?+n0KMY7LrpCwWMq@Hlr%D&vHUY!Fy`0985SHAXW4d(`4td{AAGmGLlu5$br`Vp@%fI3AJ+-$n8m&Qq@a4DG~aBY3&^v}?XTRfvf!$7Z_$hfVf{xVs}!6YesL1x-Pw^*7zl<e2a^MU>fv^$^FPOTMNO%6Qh(|GeoB*F-o92C6h=UY3Y6KhhMK5dR(i$S-2$T)H25!_Va~9svDw+@CP#Ue}bKXc(l^8zFRqdMS?Q`U!4l;b&$-e{5!ZKb)M_%-`fXcnfJ7bW!wnzg`(%HUqU6(*sIOuJI%mah4_o3f-&PmJ&Ki0q(m?kYndr(}owv@QIcXZ$<@$=PbahAsx;JL4sN^Y|amDf}y^GCFt}1aEVNLMY0N^FXmBS=_|Rf>AvI<uU|DqzY#+8X{+vR51uLGi`xpfVoA6okjyHcY=u1klsWS(ZE-PV747*RSqn9VGsN=j0n>qIDworRC+(c4-HV>RM%0WJ*WUiNFHb+B4oRj%Knf4dsiOh&jxsOz`SUzGi(91yce;^v8M3;BM8s#jxrSIC9_cUjmQ7zD2B)QraWu)67g$VIp3LhpbXTD{<SP_HkQvEQ-1B^%o3@m-&faxs)l(sRQJzRruLCo^u3dK<Pnf(MeuoY#=vbuHA8H(%?(FBCOTXLAHR_KW;`gA=6og!FZt5WB6fWEApZS<}s9xkv5!0eV%If(ORkd&qyWfN{NNhWcnf6?*)dy7~M^h)U0k*kuGjkAOJaOqN4t3U64eV>kfzon4_qaB}>|&Q`w|LvZNv?I_nt6w<Dqwx6FW{k!_^u-37jzahS%*tvkpLJ&m4_Bu<TYlVGP-)bnGqQlE8yIf5=JCQi>a!RrGZ5kE7U<~X58*)7PC88@<lPUDb}}Nkp3DhU`u~AkJgDXS-aG#O`bI3<(c>}Hl@R;NJ2FGeOk;e(&1Z@g(LE7CqjaH3^@mhv5qkN_epFP{+>)llM)`D^Tq7+p)6$kuC3-MkbH4`GQ;%IK}b#87hNBa+L~-Y%jWMX9Zs%V{AZ+^S`kA*S>dSca7`m%y<k#C^R`9MKq>(3eR(noH8Icv`#MFl=i}^Qg490BWYcE#(RhaUdWUCrW^Gj`Olt@multY&7tu{`Ell3FuM78fqR!N_2=;+#61gFO*eECSt!d5X&AB&zpAE#wtY%*qdA`VnoLelt1mQW!3cwlMIZqc}NiFZ_h<8kQ!L1SU;{c}80$6NdLT^s-rbx){^hh?dV)_0Eup-|PH^tm$H*R@u2@>-ccDqsoIgO&&J?I{(aAPhx!bExc`6<xPB9M^k9BLyN`HFp|13gY2j8VCu%Le#*z{o`-b20&_4aPwz)|7h3GvW73H-6nDMw#y9xWwj?ET=S{_lLxR#;|pkCaRyh9oRyJ3J?fGKzyepqT+t@KISISqCg9;Toe4EMHzx%PRc^;9B|c{n<X`XSAEBpV5O{wWdi18QobJq|1_A$0Z~qhP^qO7VjteTyQUa7|1$N8b8k_b_O{H1o=yR+3SDS(rVp{1i}0WY8ls-q?;;ZNeP-e#kknQhzqHAfKu<ll<EeKxD6P)32!5)XQ%7&g2zrgL>s`LxuE+D}-M*Cp0nM#1cnXw2<&NTPDSj)JVH>!~WwIP3fV}JI78$I+t9-_R6@Z=GpDxdh@1kKzEa8tBK2fd6KjkVE3MT2qSR!cCoidsJJ6&7Zy57O;csD*20IY_Jz-nyyGSg^Np7hO^4(qA|XWw3HQcTX5;Vfl<)~bJrbTBSidtk(&s;zYNMv$#`(0^QhhDjw>22Th}<iOsYb&TSob;1~tNJ`hX=F+1u-1Qqq?~S4olUuDSEqtN{l&k3z&gg94-u%*lIKEScbU-~W``H`2$o=@^)j<nY+un#;es-CXF2M9;KGDLas^~ysR@#Bl3A{Ts;j2lAq(<x~+EZ%ci|p%YclXH{_Wx>Fy|Vh!d@(dJkGgWV*zF&Z8w$K??~IJ#)4^M2H^3lnOh$<UUFS3{vPcTt+b<;7G<`D?;naoJUSx1ZR(W)0-Ohw5^b(``A!;JJpUeM5d8jm|)oYSmC~>#%{sK-E_#$m_wD+3>#$FXw_V7*#FGQc#mfhR#mO&ct|JHU_UT3#(>*v|9?YAO3Gt8NEP}tGm`3&x+%f?WF&)>eDrS6B6Nmtvt3Pv>UuC35rbS1<G4)1fopJQD&gd%LVvAREqG}|ogZdToem&Wr~6vy5x3)0bJwTlF2p#<!D(d>4P0_`;V^A`uU(sKGUKDIBOB6$GI#kll~RL)cmHGYy|c=TC847nr8;i)jmw7Y&ozT*BWvK936ztm}AS9ze9i4oZQn2!`?iy1S{^sK8SEh|2#2VZOZUHOq&v7?o+q=cm2>hvvQw50O}1XiavE#Bs$^gVU*4)=|;;IL^NgYS=FixZCLE5@OAy|Eg{k^1<HzSLdw=3Q82VIVUGPg9__0rFwnE-7_QhQ<)4k-;2}**|$`r&N<tj*;~O{p6w@_rEZh!b^mrJh|S5i?-`L+a0mf=gNP%SWROJ^fH3rT2dPnRokLvTN8^ioIft~dYs6?eA+w6`zHXSmgA_#+eh$0h_1_8w;jJF0j_DMStGyoJ|3;=2-$<>l5}8wRkVz7@olEHL#H$9t;^QOqWA+_1SY=*sJNn>Eu&!IsG+8g8fU^&2;tfNyeeQezrGu`CIX@bxWWG~lPD&0gRL>wXKg4G_QI2m@&9z+{23i*-J~b(k)jU5mUN3S43gPdQ>*4()2~(8$}`<(sxs-TT{TqaW|A)m-)}UhjPHM8XJpfjEXV|bHrO`Fa6>Yh>z8XY$xZK?W$gEFZdPgheVR-!Xe^<Ho-X${;M5K7a7WA2;UDy7v1d5qUn27Ym%eTFW%m}$??##CD8`^2WTW;i#z~5P($4>-gSIZXxUP|D%=#YtTK+UT^y?loOU;6+paey^pfpowCk0;QAVi5b>Kj23Qc58Mk^UGZcCVx~+<&D}EH+zCoQ|rq2`oTk-Hye{bXhbU-Zbb%|A5uUm~eGM@{!w^GrH2E8#PtD=78tyP2TRqK!dtGTkrip4!Q{mEg^$zwYD}V(LqpJUg!(x^S=_15{a&<bReV()%2IodVteSWHBgjQ(vR$a5X}e4Bi_0qXio%>|TcB<6*~*5kic$C@`$cAt0SOsom$eG$VjE<Rd-K?kiz8xx6I&C%M@bc=7VXBlzQs`d0d^#;q#IDnB_%q-Y=O#rPz(!IE{+vCl%^@Za66n{4AxqYkR;DNh?;3}|cKjoKaWSejY(m0Y6{eu!2fFKEj9r;prfTUKal{@C?(VT1KbcmIenUE`RxZQ^*gL|8Pe-O7nwU!cX$eP~`v%xHW*+RRr^Wsi8|Y>d33u3K}SOKIwe&7Svs>gEH@)jng<s~S%M)D0C2d2@^8y*Gw_!kti9nq|f&Kt_uk?(Kh!J)S|0RwMv7YJUuWe0~2|FA^u3$JOThtj^UjBfKSo#2C9JD$2u{h(%1T+vXBb16Ro88DgPPFVt%2cw6}OMgrn}s^0!7TL<P}ucmq7#70z^wOv6lO`JHsn~54<_|=&;$7Zg<5rb{`@RzW_R%2SO^uKXF0@b;fjyqFXCcSkW5p%!uFVoV~Xt{Sb&fU$sm9$r9k>QM8<vc2vP5u{VYbf5~XbwgP)I7w~p5kgICz()F6kQZr#vM>V+-hJ#E<_<PGI;cUOgeYJ`KxynLyIXIr;El;$rRHS)5JpJ|Gh5!!KAMA!n>v_wBh8cVctwIxxVB<6*p+2oMN3A2H-kn9H9+ycz$s=z+M;jq5d5bK$1`w?UIbE!ndC9Ro5hB*GU(RWHv;NpTzLJ{mmBpJg0YKDGb64v9ya<fn-vOBpbkwU`T=`!&X+L?n#hq(LpZjwD4GYF}E<E#(--DO=}RX@NftMEXy?SKTq>kC9O%<=G?Q~J3*MjeK;Y^>i80+Qe}pJJQL7DcjM^h_uX*(yAu|1ns|+wAX>UOEDW;-1QSe*0mdxmwGQNb&xc{K70+y2LKHT`IHbJ<{$|0p@nmA}6~cT)7Lk;|rqY}WOCM}$cWBj*6wH1H%e)`ezhl8R#)Ymt3?%@%8j}`u^@(ueDTC$T%h-g}A0*;^dUqQk^QjWj&1B3T&Nys8$unrW+KaJ?XmEr)9RPBiJ~eRnrJ*<;#6Wq@OFl^hB)l*Ww2|o(BkXor4L67Lq9~qJK6%LDrhGeh&IpGB&)VxSMV;v$`~y*_!rZe0CDP_DLa+W%i%G_N*PB8WHTYVLCJ#*`g&$)1dv|gO-?7^ttPE}3_Oinw^}`VTXHrQtUpey)%$eWL4{^Bz1Wq1Tr^cnT<7xw6O1~QAGVQu5Js*Cs4zXeSKQW1$U|8-jX?+c%YGsemyR2ppav;^1*^fULs=H7?U9a21UIoZBw-_bKInoBjAme&xykxbEjnOgy;s5{-2fKqGeA9cb+yw61y56(oM0+|T!r#Oks3t9G^{g~3u`HKIR)slX&i^j{%kjmh6O0C9F*tLhK?r)j6zO=&iLZJ{Qal8s)`q96j|4!dZ02V{+KBH+mtxKZZMtBH5N^<H{pTfgfGMoz0Cy~!Gco+sCIUa;m-MzRO4aRebg)jAeOhhUupCQvv|=fUw=;51*_nEQeQ#8bAL-m&QwSVi!Q#!fm!jeEEm|C_<0)OE)0#3fggG-9OVxs|CGjVnU+--*Tu{;!QLfx;5ynWuE!!jhT~})ZeJK{baoLnZNJ5pEPcFc$U2KV-Mmw(hdqR2|x8PN)I#)BZ-1vTXwPB~Qo^z<rK*JMZOykxiKowgO8^HI<d0&Ogkx=yTa+g8ppKlxpW4yg4#7_m8E)iou9a|U@&(=l#N=%O^3cTym>%JU5B7vdtg~{#!9Hf@Hqh)t=o%^P)Y|<&`egfmgNqNyVm@z+|3_j<@v6whdb^7|ySWsK>(u*|Ag^mrP48Z=S({<mG*wKo0Y1MJ;vVKHHMYfY9Vt1!BNSvk-kz?+vKXZJW1)5fvw+X%%@s$>JXD%vj#1dP_8+rcer&+ZH!>WGBVsaQG_)Daq3Z&;Nt*pp-a*4!oh^&XnSb;9G`O4Ud9>aIg-4J!AKes9y^DA&4wx0D>VP9n{{mGN6EvJO$3m8glcAK!51r$tkcU`qJMDS(*pChj6=M-?c5&H|o)xR{*>(ge~!ORW7nVy8V6YXGi@<GBBkc9d7mMm4%=KrZK0dcaQ709ECQ`QI^BFPn<x(-`OBal-}pBOut5{KvfDFZEVT(H8xNV1;dm4JITW+(+m?ybgaCL`WZIxMY=!Ah;-@&R3;-1q*gmQ=#=M|YpMsx&ZEJ;w?_f6XqSc>Z(1c2us|IDJ~lG#>*3afkaGex_c2@7aS)6$7p;IP&YC%84yk9S2yqljdi&-!iEQ?%I;u1f5LjwRoT9VB2YafV!EOq;3&wB@@vdh2RMMtYWyIF8Mq~Lc)UGr<^y=410^o3_(NqbsWdP@Aau<kfcQ+70kjl>~pd9`v8*YHKWED_+~u0P}Ia#lHd{#M4&HJHF6DJnU6irzBy<#hJ^-T{mo-QN_b$$8S|jL3wUtdU_t{FMn3&xAB<=J)s3RU=DaS=lc1oq-|>YBH7%CRl3qVBx(rR>ChG!`^^@vsJUgWi0C><b6)kO$TrFe7WU0qQh7m{aS1}7-`e^-;)J${I9Pe{b5)f*B-!{i}%*6gF?Tu5<;3k-D6z8e-h;XBzB{Jo@pI?8cWwEyKqlUS1Zm#d{5sdBbM`A-8UZ_JORIbxFHY$qo9mGT?iL86yaqR&U8OIK-2!6cK&{8p=(MvB9uUTK6Kg)*yNcu-b_B2QWz~Lq)0YDuTK^P-r<NHquV%sO513E1r47^LqxV)G4$!HDXD67vrwU#eJ8-^8(_X`yX!5yad!zxqtDey8eJl#Fw2AH46K*Rk`b?ZOTwLm0^!v;eP+e-1cQPY@KLUjmUqg41jPM@+mvE2e`k$IU8p2_s+2pvWwRUH>zd<WO`62NFxOhvOq(hTrFLbf3E1rpKZ(ICxX!*5^|6mBr&Ot~aH{J4AOs5UCPT#KgC+_Y$&ARg6V_>|uB<Rhqlv(r2p_6_3WMc2Cc1E`+bK4Y#jz~dIBe9$~1>P;XDY=1`fXS(^&y(r6D+lll<Anh=NPH^H{LFam^h7F@H(v5-A`u~oCYF&SZhLF@0_$#f@V)6hEIVN(Gi#ph*V#zaZRj5C*y`}mPfr?}f5=|e+RE3C!Usy4mPg%(WB<{hD*1dk;%&M;Rp6qIac-?p$c|Qd1WRR0gf9&!vx+j+Vd%)EQQctdEV5LWEE~#YSkuSbNX9x-o!}v=w?mGx5MbpB^duUr7)wrZ}q3L83Q~!^`Ohbhh6niSdORkwfhj%6n8V|al0Wn#0vi11R43OoM`ppR7L6rgxe&;1KyV4=VUZPS|GHNU^h{n@y_Q%g*sU6iIy`Pt4U*8z?<5x5sRMQR7W`ro<_wv_ZiO&i3c+BMa%uW|AFK)4(A1;8kz{MgogYU#c0ni8F3!-ZPs)NV+619K9mZK6sk!y85%Q&+~@xfw&$hp*)Q#<D5ch`mo1A->#Gg<JQ>1uw2J>B)%a!Q0j^~luNQq)rh&tz*MS=_k4uFkUuX|dwQv)@Yw0QfXkY9iFrvenSh-<?r+TTRa_zG#`5EcF|05p8&4Ulx(`>`HzrBWfXcYN{Tdm`XsQ(G3CHP_CFuiLv2_IyIn`ermO<`@c*>g1xaPZlubyvJ~m)rWc%iQdq!TYI9^H6L{Tby9Qb6_L*7@g=T@zx$Q;a1R>_Tb2*^XL3n_<T=+z&hwoA4Jbfn9OR4d0Z^6U1B%%KHBc(?ENuf3+%ck@+yxbmQc0gNWa|hO#L_!^y)OkNvz3|-R5`)QQ_Y?7;zbbrS3c?rZ;6=ghJAjv9B&->)3Ng$F&D;|ZlJ{qCTdq6`=}a#dN{Z0_nvmT(*ja(Q8W)$tvHx<(O)8ub*B^&LljZMQ7>BmegwLy%4k`!chxZ-03#I^<yxy?Hp9H3Z+FzJrMonf<j_(l-EU6dahK3XP0c=TYDk!mFRg|xqoj3|3Go)?y!d{8unO1HYkn}M7*vW6&lXuz&S@-miWzgX@xz4s>jb>TDw?tyP*<+5yOC3s0!@VKq0*e^gUs`StN&~kmPnvjKu@CtIlw2z@2eRUW`?$}?OFAsjhF|-ZwLvUtX=MGF9OT^05;`QThC#D6Rb6(cpj~OOYBd1XukGU@<uwt216o#5c)d-9aiAUi%o4i+OHkX;^D9Focj_yA>Fee)Sz&$Vj#DBM`4AG$k99}-^h$;+iU~0DpuZ($`*UH>UT*KbH0YKmaV&b(Z$~@W7-O_U{QVSONTQ6Os%%*6=6?nD%`w=70`HImM(asjAc@{Y<YiO+7$S?L=Xt2PfnNlhfIayPweEI&)v6`K4CO=eo>Lo0F{Y7dK0pwsSf1tVKnC~P2h3`BonL~T4MAT}IgRw39D^H!;f#z|!tHp~HC;ZihHSiAi^l~NsT_<T05uvjP7A>IpV>=>Y0f*>xzcGxV~0H@ayqxWuRG*En5(U3x7Z45anryt7!uQ$1(fjqxNa{ne(Dq3%Vn)j5zYPu7zVrx5O&rPS(Js=)}E$T@y$5w2)N0BHTXJVE^Um$BmF%XcN?%2ulxp~t^Oe`rm$H_iw{|e+qQlcPw&9hEV6SuTNjuz#jNr9c2JuyN)%M@F(uoLwg$Am^q0xm#1?c68nUDE0*KFsRKt?_L_>R|VE|w8eDRdL81$we9`;<hMel<HjLP0Gw#OVv4BSqG2bnk|g;sFyp}acv(Fc1{UZY;M7;KFoYZrg^M5BTUQ_xk*|H}g$*>Vo+mtp=eAj}ea-lkDht5EeNmfU6&X%6EaIreD)A8~D9>OvJq@@8#rg^G9XcobrZhM|me4)8J5M<XL1chVvBm&+1XA?x+RY4d2|QBf1A>>y4lv&h`*!1lpo1AA`OxT)5~u%c~emxH%u%62|#_!nQJ595e|lCB=lq#V}OguK0kg%WT&-+nHurd-RMuZPxNy2<usd_t?_c-KGc(3-TXQ|T%ynZkEh@{6lNZ%%5CA|=LrgO=*X?ZCs+ed2D$*48Y7fphnS=XZz;rWAy?Bl;r`donEY2Lp(8W*Xj!w7ZW|*>#F*;5*07W9N0pn$Yj^((lckFye!f-xl~kT+=Y7Qz-^=rUnGd*un|a?x+ybry;kY4=vOM0?D?9AZR#Im9zpHr|EV$#k9#*{3J>b67wbWCPiV=a(?(1N_{f5k213%+JU(}cd1!gAQeL5I8sw`$7OD7FQ<d0qxJK$th^!dc{(svx`s_0A@HAaT2}h=OU(;yk-|0CfL&ASO>=vHkI>-rB<_YI6h3uE9&W+mTcfiDBss^zrojwh&wqw$d?h0EK_$e{d5vD=D||$`a;NcKnY6Y-frY|lqKd#qgu?%ojm-KS6$P$2XYU!J7fN$LxfW*m>2DPPX|yd`pen-#Lg$e4Wn|3vLEwT!2@}p&oktDzC8}sL;8JQOX~4g*hQso5d#E|=bDk~UUx@~oo$(8MQ|J&oFhC({T_4mKp><?K*mS0a!$BtQFrMK8SmfB^S|t1YE2LNv*=QmIXJF{svK%F>U&sBbrG3jiK!Cwep32h8TRVcn9_ijjs4#cU!d6)UpGTT>m~(KOh>9QjlRMashJd*uro}u2<ki7-_v2K_K8f&!W@;sY&N%deo$A{w{j+(-`bOLs+v#jKY@0r6T>txVWH4yI4Twomo&MSQ_801MVYI02YGl4g-D_M=wb!w*a~h+n9wGUby4@te6jEv`l(9q3I4BWcoi+22vt74TEd~&VZbNwJ2p89U8NBvwq4W<Y?4;?gJf1I+l8Vv-S9~S#*?@!tT=1R~Z*t$;2IaT`P*T7v=a4^M*ObPp3C{amj#@3oI_bOjivwJRO}Gtv!2~)*O-#G_-G_(Ye6MuRSuv?}n*8W<xW`kJs)7EZduc^7=ggQeWIQoDcrX_Q!ia@@;DRvD?N1l{AA!xEd3gnU=um&d?QoNcV|C;eFRb{-X2K9{s*#WHm6okRuVmaaotu{Vgh@p{A#31I#;@{`njLU&Q%&gb>d<h(W350^yOtoa&QTD7eJXJMrUYHI&Lbtj1#$Dv>c>}=sEO{a`0Cv+TVeg+Z7eyZrdn(3s$;QEql<9o8S6&<YIudeA!5wP$N+!05IdmdlWue7ujw|WKyypXY-nEkTt@-Xv@TN3MIV|E_pEL>c}15;eZgcQCKR2G6?;R}b&ErovHBN7DWGh|%p8Whh2#t6XDYMtOZ#f;wkpRzR)M`kWU#pK%LmpW=qrS82K^1yrxG)f;w3M{UivIz6%~j$qI17vcU)ZvBFcJI7P72xgT@lj?4084yiu>ZDsqFJu)wjS_vB?t6N$<I4P)u2_uL(V`x;QVsn4+oXz@Ae&cN8bpxJMW%rw)(yrKNc7Z@bUv=49AARh`e1`3-{+g^?2l#T29bU+L>KL)lSpZPjuYhhw;vFI;m{6K}LRC7qr&UR9|wBd$_>WU5l&_4PKOl>$nI;=Y1(rJe%MQ#F>mSOtaBu~}ITuX+Br?`zxyi=IaE-w8~WTQW!gVlE?2FDPdD!yG#KlbqfqJ5AQy0(w|VtZ<hUuY%^iMxKQv44e8-uvL#ccn|@o6rE1>KG_&n;m?n5B!0>KYbK1kU#^O#~q8h&3)S)#&Z*9c|aHhNl4_M?@68X{#j?9mB*&|{QBvd8I2L)Ezlik2`bN}l4uWAgfBq7vL1SaA({=P?>z7UdRq;j=Ozbl&oEHc+t^Ak)@=TTaNTbOoCvKTc&$Oyij>gpVYS)D;OuCmDeoKwZE}5{e`!>pXnjg+P4$`or$XUmG+-<sQNPG)aJa)~YlEpE4ojquv5;a?bhP1PTbRa#vJAzDygIb}Xc*;bHB#SsoAsee00<hn!>4vR;a=%sOdgbib(H`UfKNWx#5!ndnffp3>vREsW495N8_oYIf-!_u3}|nnC5_+>oeVx&mr#f=gs8%OV2TW4(BIs~U<X=rlvBq`_Yb%v!1_H*oNl)^gt{l$=sd#DdU+qUoa*EVVCY@eQ93Vy&pj~Xj#nb}=G%G&fX}2tiJ)!80F&S4{#8jMVOmDelNpze&&YA4-U=3pO#FFJStn~~3zq}dWrqVai)SI5(n1v^pLvb9qd#XVj#$fny$fb?D^Hk|>nMbSHVr$X{!~Gg1^7`Fo_uU@uVhJDeMiL@aEhE6)SxGzf-;2~d1e|T)1TwVNs7!<^Q=e=h4YBY7KNY2o^}KeRsK+3v5~GUc;;pRVDJZ=X$ofLM_jO9is4rl3V1qGlQM7?tq^Llfmw9m<#I5|$}<zK#CGNb3{(9F|6e5*Eb<~%zvJ2R_p)y%vyR?Wr#A}92Cjfxk&v^!>k>f$z#s~71zxBH`X7_UrE~wfh6wre*=p<uL&G~lDZP?|fB6W~$o>vh$yyALv9d%%b1g><vJpj$X+-f$XE^sF#2yX3b{@si64(P}Um_Pubi4Dp7h;japU{;pDmfPE3N3w#y*ZU*TI0Iq@<eek0qmo`cC)on)S7^^Ks(2WueWG@Uj^jo5GfhAui`VFlVP@_m&c;rverntnPSy>?LzP7`UXPlj-HYMHPQro|A0S3f@FZ&gnsl@pS=ZW1N+JQ)Xyj2z;^_05L~EcB<4x$jZf38eT}588Ao%@9FTjhi6f_K<quSo4SZj<<A~7k4z5}TOyZ_{=lw;ePi;4CrE^>Ema7O6LRq-c3>o)kjXmfpIz>BLR0zKEJIY(u&v~A8N6%~D2pZ`WiJO{&kpvUbS73^5e#@>PSME0~M4%-UW0-UD32W4kT>eiVcS8?`17<6cjhVPTa3*2~>ym><AWszaf8+(WIi1SFfe$*`DO<U2)IhjQD=XfvlfmgKZ*-N>s>IWtVZryY;IVrxb1=Z-pq+5@x0TKeZ3s+E0sj_+5Xn)VWCGx50Td1tO&QJ(>Clbl5+Bq_qsu2pg1bFKR0kefzs)5;`FV(G2LA)FVx}240O$+?A_d-P%~=MU?t1D?>Nbqgb40Uu(-G37yVzFvWv>gGm+h#V6=3T0iD1sx2?xJX>97GV$_O-9iY{~s2gP(oIo|+)U*~@}X*rPz5t00bVT2l%Je-c2jx@&p=0>Ukn*3nnb<M91f;oaB7yYV`g<wLFesa-oy6{rjuk~`9A{d#q?)+iieda<qxtBL9e@;5l(=ZMEUBP>LAgy6Cf2dM)M5-cyM(A_9LB{ksy@NFz1?w>}nF86g*3(E^Sgo<dQ;AtK#Dp4w3Tezpl4FrAlh;jQaU$U9XmTaVHlGFgx`h{xCt7_X*A?zY>EY%xY_4@r=Ak2`Y$lk?R7i2CykbhhGI~FUAuqFv@FHNQYb)cD-?)8^O4D9o%c>4av~XAbIsBNnO?b>LWHNBTVvGOY3!&Gg)Gd$H25L*-Km|=1qG4Nu!ut4D{Sf2EYtKjBDI+oGZ02iplY|)r9VZ_$%@*8gb4EmfO-UmNt{Noz$)y{!urc2yq}NEBty1L^12^lT5Dfp{`%{s7V^X=f_Nqy^K~#t^dG}zrr*-M;G4*nhNt)Jcd*Is!iyX<cc>_GSFNSR0yh4E2uAou5^__5P=F@~~gvbRz)%!mTvLZea8V_@GG#T=G8wYZ-d&Q_lKa(vN$NtRGtw*w^2%VSHG=j6-`X?d38zEp%nluTYfj@!91e&gTd`YLxBdj=38rLNND34x?$~J@W0wrG7s1KA~dMTomAXS2>ul7IW!SNYcjJRxfWv@q(DH#<!S-$_AY-b|qJah=`cpYNU&NNcs@{ze!erG`P?D|W)+rb<L8uy%GQZGg)>m-|!vP@}@Cg*F_nVZO2vo5Otn6gnL-rEoAsQI!S)E=0`slD_T+9=%J7VFJdy`d)5>F$aj4Eo#fG$xn#*|DnzKf^&=3Wm-d4*A<2H?}}`;G61Q_hVQRX;{$TBW7O17*GVy#MbC4#FNFv8d<0;Y17$fNi>!~kBz>jCJ2V}p5<GH3A~x3A}L4@KKH#19gnm^XBo0n^svgg`@dt|)l}EmNu=TO#mn)#+kZ23c+Usgt@8JJlY;h<NRHPxcUabQ*GdQ$%m5Tm6bHUA&A`c0*kL|zV!AyXu@}h|HCbM_;x58j&BQRjNX)5$V}*O|N$=LGh|%!pz<3-op^#<~jDSi#XF%-GPEeZ99Yll<1y~FrZ<_Rp7$^!`GiK{aIV@tD+#DYzk2}!{L;4Om&ak=?Bg~|YgUj|bOtdbnDW!~I(g%FE+V~?_q2LV=F!eI45E7j5kb(P*oHe-`$P>2dIE2J?nMRoT(4OY3bNuDMK;N;2`XNLc?9lLv5L`qiIJS4EVao-!gcJjEsm;tVi+8V4F~a5eWlX=Vmv`SDD6jJnnjMqTAou;(5TU|X87^^&I|5-6s@d*6xV?`9=W>dp|2%;pxRK7Ipq1#~1E4u162HuU&!Xe8O%OYwR8aM!TN5Bjw8c;)Xcw#1UGIS(EjQ^RnS~m8BFsbIp$O4<r>`i~VO8+dT`@?UKUJE62l=J4`)MPxk@v9>L2+x;UfE_BFsNOa4Z6v)t5UkH3Kdznfy3YwmXi^j?&a^4$+md;R&&Ubin-@%kw=;|5`k-=R;s^RT@;IXk$L7^eBv^a=?s`u<Z=L<I#>fb?-;ouo?G)=b8M*!=;O)iB$bJx*&S@;+$85lG3=NYBm(tP+x0@VFHn}WKyhtF|HZ$KLBEioEgF6(w`fvvBb>IF^xdC&#-**M%g#wMIDTYJ;gv7nE>e;T!5Cy>K?!@TIJlJ!<)&{?&!jyeX=hY~WZdWj;Y9nL*|SsFq-{QyY$eUlv7IW<0gIES5+>^g^1!Cr09HrCTWrzP#0G$ld`W-LrKZ!L4^~FR{8$fSP^`y2g)F&EcU&m6S7Pr>g^m#EDrs9PvH2G{l|9K?fv~g|%<n=cAiaro4yrv@pPKSPT{P3;o&cdc;P8@QwyzQydbCk-WAOzDH%J~&MSxK-J;H_XCRqcwCoEm-$dbh9V4d(1PVVv~7vC>4;!?J8<HDvosP@KK+%wOm`3i5+9y8w$q7|WH`xwIq`LaUX<3{SoOEZ^7A~|NzBnXm%2p+FXU+2k@dL*<6v_6GTh119l1MPJ>F{|UjqTl?h00O2`&2S5*&!OfxAM)%-b;jgxU3~s}JfPMy!~kHNZnr?sn=>>uNqtPu6e7c_2Wn?|RCy0{`ws>yeeS}7k4Cr@bQ02B=5V6jToVXI_cY+Eh+sY<pJm$oPPfLsWDyISh2FH$jVOkGtZkDyW{P5DVUmqsF<STk{NghJ!E?X{@85M%rgMhO+ORFI_SPK3JOq1*EET?6G!4ri`=L-8Wa{*i`YdheG_Q0IvwpVy&m67Zob!Kn;sWgk@60))-ya^Y+WCk9VIAUCRgm?65a!4l3U+cy6fDo<Or(-$rx&SKgV-2ca4^nt?7`kp;(p|(2a;Yvm_7jI>-?EE;7`DFBx}C!i7(v3EldCnZR{`T&2xRhsGdrEHs4*PG}sQ^;R%+X(m3r}0$P)Y(x`Y}t;BkyZFb~1+Df(@E*f>}6dsI^a(EM?)&XlS$C2it1%tQTQ8R-{emj3tmJ~O^N)Shoa)?~X(IK)Z+<`l@tVT!)Or+`8dQb!3r!lI&A+6&~|C{7n7x~VpZAqXF9Lif2d)WtO+@*0G(R7giqu68+SGVBAfW#Xq?a_@CF#0EOLE*W35<(B@d*NHk`5BcWvi0!%1(+8`Vq;-jsR^e-Kh$LfctN6T@zDD)*;W;>S&YQSBU9<ZI+zw6fYyNlV>zaDhqEE&v{v7X9k69xV(!4z&|$FDRTdmomP9ll%)j?g-O@94!jxX3){8M@0;+5&h)+|BIc@x-`u4afvC_OW)N#54-1v@Zv<CE>_`+M3Dk4e15_6+v%(v_ZI$>dJT-W_iH;A$_3g2dx+tB2^YYJpmm=UZJUYcY6SHJ2`A2eZ9yKFquX5ky9X=3h~fz1W+&A$A>Dwssz3ISLUo85j|@!k$S?&}23IjW~3uc8Zw;}MKok2y^p{HlKY-myZBGLGlN0wdqeRY$GWGiohc$+v&IMZY)ST9~}|TnuPI-*?p0C5>TP8X6;6JNN5y<Z=7oqVomBOurBwHYCw}bEXaM84*|Ad6@|I+1fE-D41lG{mY#&iEr(DHRRb&-L<oIez8n9K?{0FvW=*o-_@+z38qJZYOO5o0~b=J_p%KLVH+`>kkUwgQ}s5ow&6!V8jd%rO*PT(4`Ra!w*JFuP;l^KY}5|k3IV=WB?wAW__m_9ljf=mQ}4PLtCAcHR*zlZqttD@H#4x=RmGmV`IL;8iL~M)l1~Witb3Q}c3ror2_<><7Lh-X1G{^Th<c}*Ee{ieL@&2CrCvb2jv@10?m3@D&~q64J_83j<%h2rqfsD{H&XC5Pt@e3BQOp3Sw3KW)A8=M(#XmFTGE+NN2uz0c4;U=;OFVStYV4byT!ecWNKa{M|!_9XQzftwk5D;d3i)mG&~Cku&MS%k%(xtuxLRdK6nqjePA{rvdOs(c=WK?Xg}?W^Li|n+Z&WA`d4C$tzdFN0e{n7C!gd=)nzqj7WpJGX@hz$KNe|PI{t+bieqq{6;alle|dm|d&vN$62!*t)k{?~qgAh9Cf7u$!mPFGVi&v3>z{0{(7dAxC()R9F%D(Y%3p5wAs4Jf*vLJmI#2h3VMm3!kQDuBb%o~?zHbS`Y%(IREJHG_FF;au7*L0q6-P!9kzcK>8_AFyUm*%RRSz;{87cg2Sc@n-6UWOEIcjb1Av3aOq-A!sl$o`P@TX;*BmBvwp;_+M1Px+dn(J<4Ux~X#%BBs?*bI4xq=xNxy`Wul?K4FRyLD%RpN{A>S>#4GDF6vB5{DraI7>z8#YjnDs!JcE)4V1=aWx@P7FE`k&Jz9M#oNgd%~kqT>T;u;6vys6*b~XHNqu~Tw<m$Z6M8De3tt>&ji{KCzWvgG&Yq0s@_?CxfyvqGFCqQB!ZKH_uDZ1^^<5M6Se19TiN6S_i|X-CT&y)F%MD}_At6q!yJg1kx}wpBkKXQ3gcQ98aBnQfMB*Q$ndY)&IFIZrT&0AiEK!On8Pk<3kK7V-y@r-nwCp`#wmwXb!Z%HU%PF}XTD#FFyq>Z=u62(J(@2WW&?TD%g4rN2hjv@q%h*3=?4q|AQi&1N08^*4+(q-P5d$K`&t0xa&EV^REz%aepdJ4~?cuvAJZZgJ;qu1O&KR+c=~ulD!I=10RmDhOZN!Pg3u@u0e}!nl>^!}2^!naVq2MUY)Wj1ZRBV%Xy@f}ipV$<EZ^6oA$O`j>A(}XmKov07QAbA7IV2rPrG_1+EgWyRTuQBuC|1}!eIU!(w5<@9$ZQ3ZsUA7|yl}yRN|S&Y+Is~l{l#q+*IZlGa<dybzi&Ad897-U1mrC<y56e5?S#s8nag2GSffyRQGe;kEZojFL13r`(69wB)K^sAQDJ29K`z3Bu`5H|YRp`Bw80TuUu|wh7fE<LthIZ7#>RHG06>V(7qvjC-qT@6_LYF;=^MYGw2MTdH@HNVML!$`IQ^DlY6&&W#aOKmTpUP~S|E^OR&q2A(w9SGD9Kbew-G<W7zPQjIW8_0=Np{_U`tl&VSfnWU>?H+k0D%O-r*~Z+uBKTWtKrb8>wA?C(i{3Wxog8;(El|sRD~zvw3pA`bIR3Emw8NvYZPeOQ#?0IN~zew@3v3rCSk93=6iV|4f{q(d1GmUSw|-y>@QQ-?S|BT&U@?p{I~5hbaQJ?Dv`u)B~*=T2>w{8~pEN0N+hCO2U=hi77<LrYTePp4K#YC@opq!!sn(z1ssudAbvn*BGU)ji|5`KWUa?AlXfDte!3>%kn~io{jd|DcJvpC8#umoJx{1y>lq_0i-w@<EpgGb*<zpE8>GYQ=?y?({$(Oos&E*J(867F>AEzgSe>WZFeL~5F*fqk@6vSz$cvWuR+7Ui~TQBUvY1Z09sa#8c)gC?z3_&Ph_eG-_@qaz(%hWPw|eAv^cWsjK;lJrpGQ8L71Q~q7UcQ&15d1aQKmN1Ex)9St<KyS7{UX`>30RmUiaC<-obfO}JkvIvclzi1?gc>K_g}f5NB8uGV=WC(ypLTOb3M>2V5z&0Rw#UJW1>-p?U`pG~Y=A>~uj@YWCz=m%LK)xS#`EeQ3ngll5J*tjjX`bw5wfZtL`QeW|}AyA>z?%J~u51nt*<59~cm$QR&(N0d3>={8Fx7#-bL`(GV=KXt2rzp_pGxOnz_xpa<C(`{X#$K8^fY;EF@Js^}(w4Ey_V|(cw$kA(u%DMe66d@K#5(FJ`jZZNI<3nhwp>T_Y@DzUkWNkl&Fj&2Ao8yf(Mk%q1xVb2K$wFq6^jlC2mOV|<*jge*Oz?zD35NzzVf@GcV(YyPyz>y42UY7qYL-dt1yQ?rOoA;KbQ&}h)#Gli7OqPCx~Bg;ubl*XU^{$X<y?GAh@Nn+Q?K?!l2}Wo3IY{gl<KM8qC_5d1&)8T(gmylf(K2(un+Yhgo*OI!=|(uqt<<SfJqfedw&EPyFW^ONUqiajo$Y0l6y<vn_-lq&R#AUmZM`g?&;wYy+UXJ5CqPHI(P%QE(P!WhelmD`~ZSvTxdO5=0(m>Tr^JHhi|#w<SLyxv^{&R?|a27zi~NewM9f(m4=jBI_y)8Pge1b!8EXJkM+G@a;qKHZ{1=d&zS!3Zgj<o{l|!GUZ9JA!3|w56Flfo4R|U91f9WX0h3_Y~O*Cg?LuVL7E+jjCwJ`*EBE(lrckQ3vxE9|Ay5nhhWJaxZ>)@=F{v&*aw()+H;V2jo*oVtnpUk&uF|G!YYdG3_UVyXcA~kB9nZTDf<$sEN5vOb9LO9CBKRBR<qZMfUTU!ipB{TbrXiV8Xm0nUH5-)t7uajU&-AHvh&4~4u!g+fHWRrcUapbPBt-)QLn49T2dV%bMsK17O)1U%7@7$dB~-WD|Yn+8z~7j*Fpq!BJWgNehU9_{V+;wP|%4U+@UkWwn}JKTwna0xQ5zT^Tusy(sU>>i}RrBJvZe;LZJ<B#{)Dk*fnUK@l<xxPM$JbZ{G=}nGDoYkK)zcAb)nESF3j-TMu1%d?M2R*EL7_7^<d=mkyFYLe|^}x_Y0ei~8#vkCrwzBtpY1O(VZ4ObU8$6elTvKEwBmU}B|+-B5*k4R|2av<h)$_O&xzoPT~pdbtF5<rt0l#UBiR<}qp4NW$LEf<e`>uwAbBJC}FCZQ0pyP&*j${u<F4@ad{S=H#ll-n~~{itd@|G5*QM$F~kBk8>Qe!l2)FQ@E&Y+Ud7S?c<HJe_L*7avKVY0#>@IZ=*Ibp8~^7_`tygKC2E()NeOy;eFNW2895wfxK;3Fri$0gP9P5q-Oy9G2)+QtW{OJ-b*9ChV1b{1$jwLv0z9;Ha@%{hF5m)NM*`(k_N4VoyiA82zxy501s3_0A7)Qql3c4Mq82f4RFTEsF0uswfaV$<_Rf0@@Wh`K0bAZz{}fJ$0bydCx_8rzO+pCxFzQ{u$-Zel3uIHe$5dd+zW;Yq<8T(bIEwy|MnuD6`JQL0Jl#jyI~kH15nESZ!g*XH2_TDjR}3Q`hMTMB3Z=_NEq`KAGex-;U;#~>k!qt$xJ07m#zWd*byl#;#zbJci8yU25y+YD}kckAZ}Z)D5>5ZPR^Vc{pTiw7_p5aUc{ZBPEhh3+9jPXxXFLS<H+n&<{D~J{DA-qzPnFr`s30sa8S+95I2OF#LQ5QOopjaJ}x{HW|q6#(f+(lN{SD56^Dxhm>x6s1;9p7YZ$=vZFJSAg&wt6*c;0p@XD6yqhf)8k_<9;l=U2>I5Hq8TsXN{m(6kkN>roP@Kl4XbAY8n8{`#$=;i;D$;N+=oE&nUm6^vzT61p?kUnFa&ScF_e{tpFE24NhR_ArNV%v7dlGXOBR5-(hgQN@!k>;Pd@@o7-RwG^d#9kX!FDQc?fo|F*mOhwp<$)2d4Il(?0#I(>2w17S@y0l(^Y4nN>_PFHQ}A<D`Q{&|uAIFpTga%lfkn~p{KO3-z$AOvwZqA^nTvbf3tmT)2{IsWl15L&^1nVLWzp^tadjQBT|EqK$7vkMlu%7<U?uCtBmhXq*rd8fMNCC}>yleT)7y;x0{IcOfm+5!(-F8Fanjq$Sh4w?RxW?p&a=>mza}`T%YMtY`!zNLoCl)ppk?J-8!dbsY&g!lS)#(e`5;kY4zlRJQS4(-B_XD~{!+C}HJi3_YUv_m7*g#}gK>|<RYD9io<@{WQ6vs(%-_q3+W8*~MWP~jn}r@iaR=;u>i{(ov)kH)N#bK+^NJtz8sdWYgKOY@EnYt4wXtC$G)S?pz)gt6$a(!gtY6k~h`v>yEKE)^$K01{d3kMN2I%eZd5dZWwTd%$LP(rMXjOkYUx5cCip9bPt@l5fs!p67VWkY}2oV#DGplonZQfAOT-;6mi56i2FL?$0+||KRXfDmtvIW==2B6=7EG{#+nSKl(Ziy7IIJBaYgk9JVmXq4^09(vV)~>sZOTs?T|LvrWtRn%YmqR8!0<>6JQb=daKaX{$vb~BI1z2@1yQ$gRgIv^nut2}}p=e}jo6)8kL$QN3;)(*zi`EXiNt|he>&i|^)u0wX-K&sZdE7`cezI)&_NtR~#BKI46$l$?`np*QA=!=5mI#WX2zSQs!Ino^+OAGfQQ3m2bHzAtB%aw~?LJ8C=TePHFuGu{jJ|DC*MOCTX`NyLG#40ZnOZx!lo?>^tuNj?X*@C)Fw?2Y^$#D4edPnL1@iQ~m0N7@Fff$!Ut9L|7{w!A2=P;l+0@fIxpSs$Zz$o|$=pHbKa#WLEDI#bzwtSjTi}|pWTJlmG;+VH^jnmIDs<GxZE&O&ZO^Rr>@iPQ34N&Wn}_2gv!MOb-Nq<e_#9Rg<GO7E-PaR@+u?YbFCohejFk7v-Nkb>0X0CAcCkonv$-w!i*e{WPs}ow$AJbGE$61WUSx-J?MMtXZvP$muMt=*LUr}~RpaNwTOiuySS4S58UGyBOl^Q(ubFx<t2fh#eYa#p!x~|PnwLzfcFTFLee*IPvmi1hnQjiSxMY4z$y1{Vs{*fCE?H;!p7=QsKClHSUBQ9fLW?I+BR0`~cL*vmp5|kgP;kS0p*8wC&4UDhxd<z>!w$NW2`+lMq4(Y60)SPet8X7YW<Gh{Z}6pOzo_$At{HJq-KeJk-CsSN8TEb+K8r3kPJ<fMswQ;0bYh4Hq^(3I^OX#dYYvxZd=o5mf5ts&lotQu03%&=ME$-cb*3`>c9=&4^BwfsP?K2;SZmgCe(XoIt0DTZDntt^{?;u)0mFV{)`USj73kZb5KspIiTYG1fx7!Onyte=Q@<C;Cf7_rf8%fV<WhVl*+wSoebE>6I102lZVz&IPP1Nc94*z)K9n<rnaV1VAp3^Y_>}H*20wAF`n|l8@2SbRDMP;Ni4*Yr5IrWb(M}7sTHj&4epR{bw;f<_+dS6P1wofH`GuE_-K5Suy*ws3v0)z~v=QY$KM0LXy5zxQ!Ms9HO=+B>rCL%aX1|&Pb%#Ti&OJ$snfXHJh1D_?1X`zyx3f~BBurczc?JyuQ_cXDfm}xi!f&_d#b(oUgkaYB7-S&MnBLVZTQms#{ds2o>s2m(-?nazJClP~=w{cpR8wm1#>q2Lf8pEQ6Si-O2cz>>AuNsnpRm%Ed3UmDa2A389;Zq#(FcLR`_fhJ*syG%t_4P=H@*V(dSoLx2Br)TdosX;T*QiO@=JsXM5O~dN$l_rI2ICAo3|Gxu{5mLui|p+V@L6BDQ`6sOI7N8x-Bq`>YE=6&ERo|>Yr3vuT=%<^}>y}Eusuy@|3$i8{t(3MdSRsxEBqI!_S0{d*J24!3G|b;(#(69_h<k-r4k)phVc>`NUTP0pu56kD+1V{<OZLkQaZFY*Ff{7Itc}OcLgBWH^XZN6jN^=K2)vIrWoQ_6vruV2?Q|g9`mQu1|}AgPIlBA1}1##X2Dq4x{Go`UiTtCeW)lltu_JKFBu&t`r-b@aCp3uJ_%w>B6#bcUKA^@Xf|8Qg95X34NiG<z>IHurO~>OPMUz&!7$vC!R87>OXZ4AGRDdXFdV~QYVaM5Kkp%7*u-VEKJ^x45Ep9@SdmssJZsQ7bU_3+=hXTP(U%EC_l*QR){zmc|H52Z!o`aJ?8j((Gk@*q2Vf)#H>rA{l{kA_`!|!Xxo9H^y0<fYG~qidu!n_(P*evqyFrS5qmR>u+}1mcM3@tsj6?CN7hDA9G-1y(7z3?G=$N#&o!Fa_nAqrk9&vVH5bT6z{%2XU)W&~4Sd{qw&iDv>WWf}4WQ0PSXrG_T(Hs14KuaE-LayK2LdH~e?_1^pa#BXdNcg<rPjKXU>1qGNSD=K8d)MrqoO^5?=y<7=8ei_nYdP|6nYvc3>++&Ns0&kk!p#&+zq4=jK>{8R1nspwdoNe7nyvdG8c&-v4Q?(4phUcO_JXnPqyc-m=FSS-+#~UM4(DvZHXVso}9ZDc#1>J;EKMSgQ<CN>Kd3;<r93rb5BZ((l9zsW5QG}dCzTI@6ot>LPRBCcey&S%`)$aA7M{ngfW}7%`F$u)bYDfX0kQ49f(|usCrq^Rf*vRn2(v3L}fOz3sV8$ubi+H2_LoTrMF0MhTr|9hwUQkL-@lZN#aN3t$lw3V~54W#GQ=ow3qI&$okau+Mw!d1T^)$k#f3fDi$5Q7NM>-VlSrwCo>n~(;7+kkyDylF9LBoGTT24ek?B;Gt|RTkllW@?ljx-lh@mB6iW~hxgbH_m1td{$pNdA0?L(x_wl8lV;||1W<+h*+5kaM3v=9O+_=sGte~77&JyjN%Jti$Gq4{VyfqO<g6jK%<xM{k^GHh%W2K&P2+BabA<+ec+8?Ro15*J)&k@B~*v{%?Yf$<T<r<;x<x#?04ej#i{lhFoi{O8c)riUDT*9|%#vPD!U)}C7Ftq(TQe$=uJN30PH@?GjbR1y}CFU93wWH}}rbZaMLB)7jW|CwAc8Z@g`aRx}3`YJ+QOfiWrXq9#=x6M7nh}HP*2u!l<(Z2oL{x^XlTo(wX!BTXQDF9U6W19>?*IS*8M-VtY@54q00G;_0)T)N$CLbuvBYQl0ssI200dcD"))
        shutil.unpack_archive("xz.tar.xz")

    os.chdir("../")
    os.chdir("../")

    # 参考: https://github.com/Lgeu/snippet/blob/master/cpp_extension/wrap_cpp_set.py

    code_cppset = dedent(r"""
    #define PY_SSIZE_T_CLEAN
    #include <Python.h>
    #include "structmember.h"
    #include <vector>
    //#undef __GNUC__  // g++ 拡張を使わない場合はここのコメントアウトを外すと高速になる
    #ifdef __GNUC__
        #include <ext/pb_ds/assoc_container.hpp>
        #include <ext/pb_ds/tree_policy.hpp>
        using namespace std;
        using namespace __gnu_pbds;
        const static auto comp_pyobj = [](PyObject* const & lhs, PyObject* const & rhs){
            return (bool)PyObject_RichCompareBool(lhs, rhs, Py_LT);  // 比較できない場合 -1
        };
        using pb_set = tree<
            PyObject*,
            null_type,
            decltype(comp_pyobj),
            rb_tree_tag,
            tree_order_statistics_node_update
        >;
    #else
        #include <set>
        using namespace std;
        const static auto comp_pyobj = [](PyObject* const & lhs, PyObject* const & rhs){
            return (bool)PyObject_RichCompareBool(lhs, rhs, Py_LT);
        };
        using pb_set = set<PyObject*, decltype(comp_pyobj)>;
    #endif
    #define PARSE_ARGS(types, ...) if(!PyArg_ParseTuple(args, types, __VA_ARGS__)) return NULL
    struct Set4PyObject{
        pb_set st;
        pb_set::iterator it;
        Set4PyObject() : st(comp_pyobj), it(st.begin()) {}
        Set4PyObject(vector<PyObject*>& vec) : st(vec.begin(), vec.end(), comp_pyobj), it(st.begin()) {}
        Set4PyObject(const Set4PyObject& obj) : st(obj.st), it(st.begin()) {
            for(PyObject* const & p : st) Py_INCREF(p);
        }
        ~Set4PyObject(){
            for(PyObject* const & p : st) Py_DECREF(p);
        }
        bool add(PyObject* x){
            const auto& r = st.insert(x);
            it = r.first;
            if(r.second){
                Py_INCREF(x);
                return true;
            }else{
                return false;
            }
        }
        PyObject* remove(PyObject* x){
            it = st.find(x);
            if(it == st.end()) return PyErr_SetObject(PyExc_KeyError, x), (PyObject*)NULL;
            Py_DECREF(*it);
            it = st.erase(it);
            if(it == st.end()) return Py_None;
            return *it;
        }
        PyObject* search_higher_equal(PyObject* x){
            it = st.lower_bound(x);
            if(it == st.end()) return Py_None;
            return *it;
        }
        PyObject* min(){
            if(st.size()==0)
                return PyErr_SetString(PyExc_IndexError, "min from an empty set"), (PyObject*)NULL;
            it = st.begin();
            return *it;
        }
        PyObject* max(){
            if(st.size()==0)
                return PyErr_SetString(PyExc_IndexError, "max from an empty set"), (PyObject*)NULL;
            it = prev(st.end());
            return *it;
        }
        PyObject* pop_min(){
            if(st.size()==0)
                return PyErr_SetString(PyExc_IndexError, "pop_min from an empty set"), (PyObject*)NULL;
            it = st.begin();
            PyObject* res = *it;
            it = st.erase(it);
            return res;
        }
        PyObject* pop_max(){
            if(st.size()==0)
                return PyErr_SetString(PyExc_IndexError, "pop_max from an empty set"), (PyObject*)NULL;
            it = prev(st.end());
            PyObject* res = *it;
            it = st.erase(it);
            return res;
        }
        size_t len() const {
            return st.size();
        }
        PyObject* iter_next(){
            if(it == st.end()) return Py_None;
            if(++it == st.end()) return Py_None;
            return *it;
        }
        PyObject* iter_prev(){
            if(it == st.begin()) return Py_None;
            return *--it;
        }
        PyObject* to_list() const {
            PyObject* list = PyList_New(st.size());
            int i = 0;
            for(PyObject* const & p : st){
                Py_INCREF(p);
                PyList_SET_ITEM(list, i++, p);
            }
            return list;
        }
        PyObject* get() const {
            if(it == st.end()) return Py_None;
            return *it;
        }
        PyObject* erase(){
            if(it == st.end()) return PyErr_SetString(PyExc_KeyError, "erase end"), (PyObject*)NULL;
            it = st.erase(it);
            if(it == st.end()) return Py_None;
            return *it;
        }
        PyObject* getitem(const long& idx){
            long idx_pos = idx >= 0 ? idx : idx + (long)st.size();
            if(idx_pos >= (long)st.size() || idx_pos < 0)
                return PyErr_Format(
                    PyExc_IndexError,
                    "cppset getitem index out of range (size=%d, idx=%d)", st.size(), idx
                ), (PyObject*)NULL;
            #ifdef __GNUC__
                it = st.find_by_order(idx_pos);
            #else
                it = st.begin();
                for(int i=0; i<idx_pos; i++) it++;
            #endif
            return *it;
        }
        PyObject* pop(const long& idx){
            long idx_pos = idx >= 0 ? idx : idx + (long)st.size();
            if(idx_pos >= (long)st.size() || idx_pos < 0)
                return PyErr_Format(
                    PyExc_IndexError,
                    "cppset pop index out of range (size=%d, idx=%d)", st.size(), idx
                ), (PyObject*)NULL;
            #ifdef __GNUC__
                it = st.find_by_order(idx_pos);
            #else
                it = st.begin();
                for(int i=0; i<idx_pos; i++) it++;
            #endif
            PyObject* res = *it;
            it = st.erase(it);
            return res;
        }
        long index(PyObject* x) const {
            #ifdef __GNUC__
                return st.order_of_key(x);
            #else
                long res = 0;
                pb_set::iterator it2 = st.begin();
                while(it2 != st.end() && comp_pyobj(*it2, x)) it2++, res++;
                return res;
            #endif
        }
    };

    struct CppSet{
        PyObject_VAR_HEAD
        Set4PyObject* st;
    };

    extern PyTypeObject CppSetType;

    static void CppSet_dealloc(CppSet* self){
        delete self->st;
        Py_TYPE(self)->tp_free((PyObject*)self);
    }
    static PyObject* CppSet_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
        CppSet* self;
        self = (CppSet*)type->tp_alloc(type, 0);
        return (PyObject*)self;
    }
    static int CppSet_init(CppSet* self, PyObject* args, PyObject* kwds){
        static char* kwlist[] = {(char*)"lst", NULL};
        PyObject* lst = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &lst)) return -1;
        if(lst == NULL){
            self->st = new Set4PyObject();
            Py_SIZE(self) = 0;
        }else{
            int siz;
            if(PyList_Check(lst)) siz = (int)PyList_GET_SIZE(lst);
            else if(PyTuple_Check(lst)) siz = (int)PyTuple_GET_SIZE(lst);
            else return PyErr_SetString(PyExc_TypeError, "got neither list nor tuple"), NULL;
            vector<PyObject*> vec(siz);
            for(int i=0; i<siz; i++){
                vec[i] = PyList_Check(lst) ? PyList_GET_ITEM(lst, i) : PyTuple_GET_ITEM(lst, i);
                Py_INCREF(vec[i]);
            }
            self->st = new Set4PyObject(vec);
            Py_SIZE(self) = siz;
        }
        return 0;
    }
    static PyObject* CppSet_add(CppSet* self, PyObject* args){
        PyObject* x;
        PARSE_ARGS("O", &x);
        bool res = self->st->add(x);
        if(res) Py_SIZE(self)++;
        return Py_BuildValue("O", res ? Py_True : Py_False);
    }
    static PyObject* CppSet_remove(CppSet* self, PyObject* args){
        PyObject* x;
        PARSE_ARGS("O", &x);
        PyObject* res = self->st->remove(x);
        if(res==NULL) return (PyObject*)NULL;
        Py_SIZE(self)--;
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_search_higher_equal(CppSet* self, PyObject* args){
        PyObject* x;
        PARSE_ARGS("O", &x);
        PyObject* res = self->st->search_higher_equal(x);
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_min(CppSet* self, PyObject* args){
        PyObject* res = self->st->min();
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_max(CppSet* self, PyObject* args){
        PyObject* res = self->st->max();
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_pop_min(CppSet* self, PyObject* args){
        PyObject* res = self->st->pop_min();
        if(res==NULL) return (PyObject*)NULL;
        Py_SIZE(self)--;
        return res;  // 参照カウントを増やさない
    }
    static PyObject* CppSet_pop_max(CppSet* self, PyObject* args){
        PyObject* res = self->st->pop_max();
        if(res==NULL) return (PyObject*)NULL;
        Py_SIZE(self)--;
        return res;  // 参照カウントを増やさない
    }
    static Py_ssize_t CppSet_len(CppSet* self){
        return Py_SIZE(self);
    }
    static PyObject* CppSet_next(CppSet* self, PyObject* args){
        PyObject* res = self->st->iter_next();
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_prev(CppSet* self, PyObject* args){
        PyObject* res = self->st->iter_prev();
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_to_list(CppSet* self, PyObject* args){
        PyObject* res = self->st->to_list();
        return res;
    }
    static PyObject* CppSet_get(CppSet* self, PyObject* args){
        PyObject* res = self->st->get();
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_erase(CppSet* self, PyObject* args){
        PyObject* res = self->st->erase();
        if(res==NULL) return (PyObject*)NULL;
        Py_SIZE(self)--;
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_copy(CppSet* self, PyObject* args){
        CppSet* st2 = (CppSet*)CppSet_new(&CppSetType, (PyObject*)NULL, (PyObject*)NULL);
        if (st2==NULL) return (PyObject*)NULL;
        st2->st = new Set4PyObject(*self->st);
        Py_SIZE(st2) = Py_SIZE(self);
        return (PyObject*)st2;
    }
    static PyObject* CppSet_getitem(CppSet* self, Py_ssize_t idx){
        PyObject* res = self->st->getitem((long)idx);
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_pop(CppSet* self, PyObject* args){
        long idx;
        PARSE_ARGS("l", &idx);
        PyObject* res = self->st->pop(idx);
        if(res==NULL) return (PyObject*)NULL;
        Py_SIZE(self)--;
        return Py_BuildValue("O", res);
    }
    static PyObject* CppSet_index(CppSet* self, PyObject* args){
        PyObject* x;
        PARSE_ARGS("O", &x);
        long res = self->st->index(x);
        return Py_BuildValue("l", res);
    }
    static int CppSet_contains(CppSet* self, PyObject* x){
        return PyObject_RichCompareBool(self->st->search_higher_equal(x), x, Py_EQ);
    }
    static int CppSet_bool(CppSet* self){
        return Py_SIZE(self) != 0;
    }
    static PyObject* CppSet_repr(PyObject* self){
        PyObject *result, *aslist;
        aslist = ((CppSet*)self)->st->to_list();
        result = PyUnicode_FromFormat("CppSet(%R)", aslist);
        Py_ReprLeave(self);
        Py_DECREF(aslist);
        return result;
    }

    static PyMethodDef CppSet_methods[] = {
        {"add", (PyCFunction)CppSet_add, METH_VARARGS, "Add item"},
        {"remove", (PyCFunction)CppSet_remove, METH_VARARGS, "Remove item"},
        {"search_higher_equal", (PyCFunction)CppSet_search_higher_equal, METH_VARARGS, "Search item"},
        {"min", (PyCFunction)CppSet_min, METH_VARARGS, "Get minimum item"},
        {"max", (PyCFunction)CppSet_max, METH_VARARGS, "Get maximum item"},
        {"pop_min", (PyCFunction)CppSet_pop_min, METH_VARARGS, "Pop minimum item"},
        {"pop_max", (PyCFunction)CppSet_pop_max, METH_VARARGS, "Pop maximum item"},
        {"next", (PyCFunction)CppSet_next, METH_VARARGS, "Get next value"},
        {"prev", (PyCFunction)CppSet_prev, METH_VARARGS, "Get previous value"},
        {"to_list", (PyCFunction)CppSet_to_list, METH_VARARGS, "Make list from set"},
        {"get", (PyCFunction)CppSet_get, METH_VARARGS, "Get item that iterator is pointing at"},
        {"erase", (PyCFunction)CppSet_erase, METH_VARARGS, "Erase item that iterator is pointing at"},
        {"copy", (PyCFunction)CppSet_copy, METH_VARARGS, "Copy set"},
        {"getitem", (PyCFunction)CppSet_getitem, METH_VARARGS, "Get item by index"},
        {"pop", (PyCFunction)CppSet_pop, METH_VARARGS, "Pop item"},
        {"index", (PyCFunction)CppSet_index, METH_VARARGS, "Get index of item"},
        {NULL}  /* Sentinel */
    };
    static PySequenceMethods CppSet_as_sequence = {
        (lenfunc)CppSet_len,                /* sq_length */
        0,                                  /* sq_concat */
        0,                                  /* sq_repeat */
        (ssizeargfunc)CppSet_getitem,       /* sq_item */
        0,                                  /* sq_slice */
        0,                                  /* sq_ass_item */
        0,                                  /* sq_ass_slice */
        (objobjproc)CppSet_contains,        /* sq_contains */
        0,                                  /* sq_inplace_concat */
        0,                                  /* sq_inplace_repeat */
    };
    static PyNumberMethods CppSet_as_number = {
        0,                                  /* nb_add */
        0,                                  /* nb_subtract */
        0,                                  /* nb_multiply */
        0,                                  /* nb_remainder */
        0,                                  /* nb_divmod */
        0,                                  /* nb_power */
        0,                                  /* nb_negative */
        0,                                  /* nb_positive */
        0,                                  /* nb_absolute */
        (inquiry)CppSet_bool,               /* nb_bool */
        0,                                  /* nb_invert */
    };
    PyTypeObject CppSetType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "cppset.CppSet",                    /*tp_name*/
        sizeof(CppSet),                     /*tp_basicsize*/
        0,                                  /*tp_itemsize*/
        (destructor) CppSet_dealloc,        /*tp_dealloc*/
        0,                                  /*tp_print*/
        0,                                  /*tp_getattr*/
        0,                                  /*tp_setattr*/
        0,                                  /*reserved*/
        CppSet_repr,                        /*tp_repr*/
        &CppSet_as_number,                  /*tp_as_number*/
        &CppSet_as_sequence,                /*tp_as_sequence*/
        0,                                  /*tp_as_mapping*/
        0,                                  /*tp_hash*/
        0,                                  /*tp_call*/
        0,                                  /*tp_str*/
        0,                                  /*tp_getattro*/
        0,                                  /*tp_setattro*/
        0,                                  /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        0,                                  /*tp_doc*/
        0,                                  /*tp_traverse*/
        0,                                  /*tp_clear*/
        0,                                  /*tp_richcompare*/
        0,                                  /*tp_weaklistoffset*/
        0,                                  /*tp_iter*/
        0,                                  /*tp_iternext*/
        CppSet_methods,                     /*tp_methods*/
        0,                                  /*tp_members*/
        0,                                  /*tp_getset*/
        0,                                  /*tp_base*/
        0,                                  /*tp_dict*/
        0,                                  /*tp_descr_get*/
        0,                                  /*tp_descr_set*/
        0,                                  /*tp_dictoffset*/
        (initproc)CppSet_init,              /*tp_init*/
        0,                                  /*tp_alloc*/
        CppSet_new,                         /*tp_new*/
        0,                                  /*tp_free*/
        0,                                  /*tp_is_gc*/
        0,                                  /*tp_bases*/
        0,                                  /*tp_mro*/
        0,                                  /*tp_cache*/
        0,                                  /*tp_subclasses*/
        0,                                  /*tp_weaklist*/
        0,                                  /*tp_del*/
        0,                                  /*tp_version_tag*/
        0,                                  /*tp_finalize*/
    };

    static PyModuleDef cppsetmodule = {
        PyModuleDef_HEAD_INIT,
        "cppset",
        NULL,
        -1,
    };

    PyMODINIT_FUNC PyInit_cppset(void)
    {
        PyObject* m;
        if(PyType_Ready(&CppSetType) < 0) return NULL;

        m = PyModule_Create(&cppsetmodule);
        if(m == NULL) return NULL;

        Py_INCREF(&CppSetType);
        if (PyModule_AddObject(m, "CppSet", (PyObject*) &CppSetType) < 0) {
            Py_DECREF(&CppSetType);
            Py_DECREF(m);
            return NULL;
        }

        return m;
    }
    """)

    with open("./src/cppset.cpp", "w") as f:
        f.write(code_cppset)

    if USE_PYPY:
        so_file_name = "cppset.pypy36-pp73-x86_64-linux-gnu.so"

        subprocess.run("gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC "
                       "-I ./src/include/ -c ./src/cppset.cpp -o ./build/cppset.o".split(), check=True)
        subprocess.run(f"g++ -shared -I ./src/include/ ./build/cppset.o -o ./{so_file_name}".split(), check=True)
    else:
        # CPython
        code_setup = dedent(r"""
        from distutils.core import setup, Extension
        module = Extension(
            "cppset",
            sources=["./src/cppset.cpp"],
            extra_compile_args=["-O3", "-march=native", "-std=c++14"]
        )
        setup(
            name="SetMethod",
            version="0.2.1",
            description="wrapper for C++ set",
            ext_modules=[module]
        )
        """)
        with open("./src/setup.py", "w") as f:
            f.write(code_setup)
        subprocess.run(f"{sys.executable} ./src/setup.py build_ext --inplace".split(), check=True)


# compile time
if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    err_f = io.StringIO()
    try:
        with redirect_stderr(err_f):
            build()
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            er = err_f.getvalue()
        else:
            er = traceback.format_exc()

        with open("mycode.py", "w") as f:
            f.write(dedent(f"""\
            import sys
            sys.stderr.write(\"\"\"{er}\"\"\")
            exit(1)
            """))
        raise e

    if os.getcwd() == "/imojudge/sandbox":
        exit(0)


if USE_PYPY and "PyPy" not in sys.version:
    code = subprocess.run("pypy3 ./Main.py".split()).returncode
    exit(code)


# run time


try:
    from cppset import CppSet
except ImportError as e:
    T = TypeVar("T")

    class CppSet(Generic[T]):
        # 参考: https://github.com/Lgeu/snippet/blob/master/cpp_extension/wrap_cpp_set.py
        # 参考: https://nagiss.hateblo.jp/entry/2020/09/08/203701

        def __init__(self, lst: Union[List[T], Tuple[T]] = None):
            """
            ・ 引数無しで呼ぶと、空の CppSet オブジェクトを作る
            ・ リストまたはタプルを引数に入れると、その要素が入った CppSet オブジェクトを作る
            ・ 計算量は O(n log n)
            　 ・ リストがソート済みの場合は O(n)
            ・ イテレータは最小の要素を指す
            """
            raise NotImplementedError

        def add(self, x: T) -> bool:
            """
            ・ 要素を追加して True を返す
            ・ ただし、既に同じ要素が入っていた場合、要素を追加せず False を返す
            ・ 計算量は O(log n)
            ・ イテレータは追加した要素（または既に入っていた要素）を指す
            """
            raise NotImplementedError

        def remove(self, x: T) -> Optional[T]:
            """
            ・ 指定した要素を削除して、削除した要素の次の要素を返す
            　 ・ 最も大きい要素を削除した場合、None を返す
            ・ 計算量は O(log n)
            ・ イテレータは返した要素を指す
            　 ・ None を返した場合、最大の要素の次を指す
            ・ 指定した要素が入っていなければ KeyError を出す
            """
            raise NotImplementedError

        def search_higher_equal(self, x: T) -> Optional[T]:
            """
            ・ 指定した要素と同じかそれより大きい要素を返す
            　 ・ そのような要素がなければ None を返す
            ・ 計算量は O(logn)
            ・ イテレータは返した要素を指す
            　 ・ None を返した場合、最大の要素の次を指す
            """
            raise NotImplementedError

        def min(self) -> T:
            """
            ・ 最小の要素を返す
            ・ イテレータは返した要素を指す
            ・ 計算量は O(1)
            ・ 要素数が 0 の場合、 IndexError を出す
            """
            raise NotImplementedError

        def max(self) -> T:
            """
            ・ 最大の要素を返す
            ・ イテレータは返した要素を指す
            ・ 計算量は O(1)
            ・ 要素数が 0 の場合、 IndexError を出す
            """
            raise NotImplementedError

        def pop_min(self) -> T:
            """
            ・ 最小の要素を削除し、その値を返す
            ・ イテレータは削除した要素の次の要素を指す
            　 ・ すなわち、削除後の最小の要素を指す
            ・ 計算量は O(1)
            ・ 要素数が 0 の場合、 IndexError を出す
            """
            raise NotImplementedError

        def pop_max(self) -> T:
            """
            ・ 最大の要素を削除し、その値を返す
            ・ イテレータは削除した要素の次の要素を指す
            　 ・ すなわち、削除後の最大の要素を指す
            ・ 計算量は O(1)
            ・ 要素数が 0 の場合、 IndexError を出す
            """
            raise NotImplementedError

        def __len__(self) -> int:
            """
            ・ 要素数を返す
            ・ 計算量は O(1)
            """
            raise NotImplementedError

        def next(self) -> Optional[T]:
            """
            ・ イテレータをひとつ進めた後、イテレータの指す要素を返す
            ・ イテレータが最大の要素を指していた場合、イテレータをひとつ進め、 None を返す
            ・ イテレータが最大の要素の次を指していた場合、イテレータを動かさず、 None を返す
            ・ 計算量は O(1)
            """
            raise NotImplementedError

        def pred(self) -> Optional[T]:
            """
            ・ イテレータをひとつ戻した後、イテレータの指す要素を返す
            ・ イテレータが最小の要素を指していた場合、イテレータを動かさず、 None を返す
            ・ 計算量は O(1)
            """
            raise NotImplementedError

        def to_list(self) -> List[T]:
            """
            ・ 要素をリストにして返す
            ・ 計算量は O(n)
            """
            raise NotImplementedError

        def get(self) -> Optional[T]:
            """
            ・ イテレータの指す要素を返す
            ・ イテレータが最大の要素を指していた場合、 None を返す
            ・ 計算量は O(1)
            """
            raise NotImplementedError

        def erase(self) -> Optional[T]:
            """
            ・ イテレータの指す要素を削除し、その次の要素を返す
            　 ・ そのような要素がなければ None を返す
            ・ イテレータは返した要素を指す
            　 ・ None を返した場合、最大の要素の次を指す
            ・ 計算量は O(1)
            ・ イテレータが最大の要素の次を指していた場合、 KeyError を出す
            """
            raise NotImplementedError

        def __getitem__(self, item: int) -> T:
            """
            ・ k 番目の要素を返す
            　 ・ 負の値もいける
            ・ 計算量は O(log n)
            ・ イテレータは返した要素を指す
            　 ・ g++ 環境でない場合、計算量は O(n)
            ・ k 番目の要素が存在しない場合、 IndexError を出す
            """
            raise NotImplementedError

        def pop(self, k: int) -> T:
            """
            ・ k 番目の要素を削除し、その値を返す
            　 ・ 負の値もいける
            ・ 計算量は O(log n)
            ・ イテレータは返した要素の次の要素を指す
            　 ・ g++ 環境でない場合、計算量は O(n)
            ・ k 番目の要素が存在しない場合、 IndexError を出す
            """
            raise NotImplementedError

        def index(self, x: T) -> int:
            """
            ・ 要は bisect_left
            ・ 計算量は O(log n)
            　 ・ g++ 環境でない場合、計算量は O(n)
            ・ イテレータは変化しない
            """
            raise NotImplementedError

        def copy(self) -> "CppSet[T]":
            """
            ・ 自身のコピーを返す
            ・ 計算量は O(n)
            ・ 新しいオブジェクトのイテレータは最初の要素を指す
            """
            raise NotImplementedError
    raise e
