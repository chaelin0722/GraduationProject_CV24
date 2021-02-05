package com.example.server;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;

public class FullInfoPop extends AppCompatActivity {
    TextView SITUATION, DATETIME, ADDR;
    String situation, DateTime, latitude, longitude;
    JSONObject jsonObject;
    JSONObject Addr;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_full_info_pop);

        Intent intent = getIntent();
        String data = intent.getStringExtra("message");
        JSONparse(data);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);

        SITUATION = (TextView) findViewById(R.id.situationDetail);
        DATETIME = (TextView) findViewById(R.id.dateTimeDetail);
        ADDR = (TextView) findViewById(R.id.locationDetail);

        //각 UI 요소의 내용 업데이트
        SITUATION.setText(situation);
        DATETIME.setText(DateTime);
        ADDR.setText(latitude + " , " + longitude);


    }





    public void JSONparse(String jsonStr) {
        try {
            jsonObject = new JSONObject(jsonStr);  //JSON string을 JSON 객체로 변경

            situation = jsonObject.getString("situation");  //위험상황 변수 저장
            DateTime = jsonObject.getString("DateTime");  //사건 발생 일시 및 시간 변수 저장

            Addr = jsonObject.getJSONObject("addr");
            latitude = Addr.getString("lat");  //사건 발생 위치 위도 저장
            longitude = Addr.getString("long");  //사건 발생 위치 경도 저장

        } catch (JSONException e) {
            e.printStackTrace();
        }
    }


}